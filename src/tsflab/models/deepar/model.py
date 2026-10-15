"""Clean-room DeepAR implementation derived from the likelihood factorization."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tsflab.models._components.gaussian_parameter_head import GaussianParameterHead
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Global autoregressive LSTM with an independent Gaussian likelihood."""

    output_type = "distribution"
    distribution_family = "gaussian"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        label_len: int = 0,
        features: str = "M",
        embedding_size: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        cov_feat_size: int = 0,
        dropout: float = 0.1,
        checkpoint_steps: int = 24,
    ) -> None:
        super().__init__()
        del label_len
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be M, S, or MS")
        self.checkpoint_steps = checkpoint_steps
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.cov_feat_size = cov_feat_size
        self.value_embedding = nn.Linear(1, embedding_size)
        self.recurrent = nn.LSTM(
            embedding_size + cov_feat_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.likelihood = GaussianParameterHead(hidden_size, 1, eps=1e-6)

    def _covariates(
        self,
        marks: torch.Tensor | None,
        batch: int,
        steps: int,
        channels: int,
    ) -> torch.Tensor:
        if self.cov_feat_size == 0:
            return self.value_embedding.weight.new_empty(batch * channels, steps, 0)
        if marks is None:
            base = self.value_embedding.weight.new_zeros(batch, steps, self.cov_feat_size)
        else:
            if marks.ndim != 3 or marks.shape[:2] != (batch, steps):
                raise ValueError(f"marks must have shape [batch, {steps}, features]")
            base = marks[..., : self.cov_feat_size]
            if base.shape[-1] < self.cov_feat_size:
                base = torch.nn.functional.pad(base, (0, self.cov_feat_size - base.shape[-1]))
        return base.unsqueeze(1).expand(-1, channels, -1, -1).reshape(batch * channels, steps, self.cov_feat_size)

    def _decode(self, feedback, h, c, future_cov, native: bool):
        """Roll the decoder over ``future_cov.shape[1]`` steps from ``(feedback, h, c)``.

        Returns the next feedback and state and ``[series, steps, 1]`` locations
        and scales. ``native`` runs the LSTM without cuDNN (see ``forward``).
        """
        previous = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = previous and not native
        try:
            state = (h, c)
            locations: list[torch.Tensor] = []
            scales: list[torch.Tensor] = []
            for step in range(future_cov.shape[1]):
                recurrent_input = torch.cat(
                    (
                        self.value_embedding(feedback),
                        future_cov[:, step : step + 1],
                    ),
                    dim=-1,
                )
                recurrent_output, state = self.recurrent(recurrent_input, state)
                location, scale = self.likelihood(recurrent_output[:, 0])
                locations.append(location)
                scales.append(scale)
                # Deterministic inference uses the likelihood mean as the next input.
                feedback = location.unsqueeze(1)
        finally:
            torch.backends.cudnn.enabled = previous
        return (feedback, *state, torch.stack(locations, dim=1), torch.stack(scales, dim=1))

    def _future_marks(self, x_mark_dec: torch.Tensor | None) -> torch.Tensor | None:
        if x_mark_dec is None:
            return None
        future_marks = x_mark_dec[:, -self.pred_len :]
        if future_marks.shape[1] != self.pred_len:
            raise ValueError("x_mark_dec does not cover the forecast horizon")
        return future_marks

    def _rollout(self, feedback, state, future_cov) -> tuple[torch.Tensor, torch.Tensor]:
        """``[series, pred_len, 1]`` decoder locations and scales from ``(feedback, state)``."""
        segment = self.checkpoint_steps
        if not (segment and self.training and torch.is_grad_enabled()):
            _, _, _, location, scale = self._decode(feedback, *state, future_cov, False)
            return location, scale
        # Recompute each decoder segment in backward: with autograd the full
        # rollout keeps every step of every series alive (about 50 GB for
        # 16 x 862 series over 720 steps). The forward value and gradient are
        # those of the plain rollout; LSTM dropout runs on the native kernel,
        # whose torch RNG the checkpoint replays, so recomputed masks match.
        native = self.recurrent.dropout > 0 and self.recurrent.num_layers > 1
        h, c = state
        locations, scales = [], []
        for start in range(0, self.pred_len, segment):
            feedback, h, c, location, scale = checkpoint(
                self._decode,
                feedback,
                h,
                c,
                future_cov[:, start : start + segment],
                native,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            locations.append(location)
            scales.append(scale)
        return torch.cat(locations, dim=1), torch.cat(scales, dim=1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        channels = self.enc_in
        history_cov = self._covariates(x_mark_enc, batch, self.seq_len, channels)
        history = x_enc.transpose(1, 2).reshape(batch * channels, self.seq_len, 1)
        encoded = torch.cat((self.value_embedding(history), history_cov), dim=-1)
        _, state = self.recurrent(encoded)

        future_cov = self._covariates(self._future_marks(x_mark_dec), batch, self.pred_len, channels)
        location, scale = self._rollout(history[:, -1:], state, future_cov)
        location = location.view(batch, channels, self.pred_len)
        scale = scale.view(batch, channels, self.pred_len)
        location = location.transpose(1, 2)
        scale = scale.transpose(1, 2)
        if self.features == "MS":
            location = location[..., -1:]
            scale = scale[..., -1:]
        return torch.stack((location, scale), dim=-1)
