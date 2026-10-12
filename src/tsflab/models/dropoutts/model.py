"""DropoutTS (Zhong et al., ICML 2026, arXiv:2601.21726).

Independent implementation of sample-adaptive dropout as a training-time plug-in
on a channel-independent PatchTST backbone (the ``patchtst`` component). A
spectral noise scorer turns each training window into a reconstruction-residual
noise score (Sec. 4.1, Eq. 2-7); batch min-max scaling and a learnable tanh
sensitivity curve map the scores to per-sample dropout rates in
``[p_min, p_max]`` (Eq. 8-9); every ``nn.Dropout`` site of the backbone is
replaced by a straight-through sample-adaptive dropout (Eq. 10-11), so the task
loss trains the scorer end to end. Inference runs the plain backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.patchtst import PatchTSTBackbone


class DropoutRateState:
    """Per-model holder of the current training batch's dropout rates ``[B]``."""

    def __init__(self) -> None:
        self.rates: torch.Tensor | None = None


class SampleAdaptiveDropout(nn.Module):
    """Eq. (10)-(11): Bernoulli(1 - p_i) mask with a straight-through path to ``p``.

    ``M = B + ((1 - p) - stopgrad(1 - p))`` equals the binary mask forward, while the
    backward pass differentiates ``(1 - p)``; the output is ``H * M / (1 - p)``.
    Rates of length ``B`` are repeated for channel-folded ``[B * C, ...]`` inputs.
    Without rates (evaluation, or a call outside ``Model.forward``) the module
    falls back to ordinary dropout with its original rate.
    """

    def __init__(self, p: float, state: DropoutRateState) -> None:
        super().__init__()
        self.p = float(p)
        self.state = state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs
        rates = self.state.rates
        if rates is None:
            return F.dropout(inputs, self.p, training=True)
        if inputs.shape[0] != rates.shape[0]:
            if inputs.shape[0] % rates.shape[0]:
                raise ValueError(
                    f"cannot align {rates.shape[0]} sample rates with batch {inputs.shape[0]}"
                )
            rates = rates.repeat_interleave(inputs.shape[0] // rates.shape[0])
        rates = rates.view(-1, *([1] * (inputs.ndim - 1)))
        keep = 1.0 - rates
        binary = (torch.rand_like(inputs) > rates).to(inputs.dtype)
        mask = binary + (keep - keep.detach())
        return inputs * mask / (keep + 1e-6)


class SpectralNoiseScorer(nn.Module):
    """Sec. 4.1: global OLS detrend, log-amplitude spectrum, SFM-anchored soft mask.

    Learnable parameters are the per-channel SFM scale/bias of the dynamic threshold
    ``tau = sigmoid(w_s * SFM + b_s)`` and the mask sharpness ``alpha``
    (``Softplus(alpha)`` in Eq. 6). ``forward`` returns the per-sample MAE between
    the window and its spectrally filtered reconstruction (Eq. 7).
    """

    def __init__(self, seq_len: int, channels: int, init_alpha: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.sfm_scale = nn.Parameter(torch.ones(1, channels, 1))
        self.sfm_bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        steps = torch.arange(seq_len, dtype=torch.float32)
        design = torch.stack((torch.ones(seq_len), steps), dim=1)
        self.register_buffer("steps", steps, persistent=False)
        self.register_buffer("design_pinv", torch.linalg.pinv(design), persistent=False)

    def trend(self, series: torch.Tensor) -> torch.Tensor:
        """Eq. (2): least-squares line ``T w* + b*`` per channel; ``series`` is ``[B, C, L]``."""
        intercept, slope = (series @ self.design_pinv.T).unbind(-1)
        return intercept.unsqueeze(-1) + slope.unsqueeze(-1) * self.steps

    @staticmethod
    def spectral_flatness(log_amplitude: torch.Tensor) -> torch.Tensor:
        """Geometric over arithmetic mean of the power of the log-amplitude spectrum."""
        power = log_amplitude.square() + 1e-8
        geometric = power.log().mean(-1, keepdim=True).exp()
        return geometric / (power.mean(-1, keepdim=True) + 1e-8)

    def details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        series = x.transpose(1, 2)  # [B, C, L]
        trend = self.trend(series)
        spectrum = torch.fft.rfft(series - trend, dim=-1)  # Eq. (3)
        log_amplitude = torch.log1p(spectrum.abs())
        low = log_amplitude.amin(-1, keepdim=True)
        span = (log_amplitude.amax(-1, keepdim=True) - low).clamp_min(1e-6)
        normalized = (log_amplitude - low) / span  # Eq. (4)
        sfm = self.spectral_flatness(log_amplitude)  # Eq. (5)
        threshold = torch.sigmoid(self.sfm_scale * sfm + self.sfm_bias)
        mask = torch.sigmoid(F.softplus(self.alpha) * (normalized - threshold))  # Eq. (6)
        clean = torch.fft.irfft(spectrum * mask, n=self.seq_len, dim=-1) + trend
        score = (series - clean).abs().mean(dim=(1, 2))  # Eq. (7)
        return {
            "trend": trend, "normalized": normalized, "sfm": sfm,
            "threshold": threshold, "mask": mask, "reconstruction": clean, "score": score,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.details(x)["score"]


class Model(nn.Module):
    """PatchTST backbone whose dropout sites use DropoutTS sample-adaptive rates."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str | None = "end",
        e_layers: int = 1,
        d_model: int = 256,
        n_heads: int = 1,
        d_ff: int = 1024,
        activation: str = "gelu",
        norm: str = "BatchNorm",
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        revin: bool = True,
        affine: bool = True,
        p_min: float = 0.05,
        p_max: float = 0.5,
        init_alpha: float = 10.0,
        init_sensitivity: float = 1.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len, pred_len and enc_in must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.backbone = PatchTSTBackbone(
            c_in=enc_in, context_window=seq_len, target_window=pred_len,
            patch_len=patch_len, stride=stride, padding_patch=padding_patch,
            n_layers=e_layers, d_model=d_model, n_heads=n_heads, d_k=None, d_v=None,
            d_ff=d_ff, activation=activation, norm=norm, attn_dropout=attn_dropout,
            res_dropout=dropout, ffn_dropout=dropout, proj_dropout=dropout,
            head_dropout=dropout, pre_norm=False, pe="zeros", learn_pe=False,
            head_type="flatten", individual=False, revin=revin, affine=affine,
            subtract_last=False,
        )
        self.rate_state = DropoutRateState()
        self.adaptive_dropouts = self._replace_dropouts(self.backbone, self.rate_state)
        self.scorer = SpectralNoiseScorer(seq_len, enc_in, init_alpha)
        self.sensitivity = nn.Parameter(torch.tensor(float(init_sensitivity)))

    @staticmethod
    def _replace_dropouts(module: nn.Module, state: DropoutRateState) -> int:
        """Swap every ``nn.Dropout`` submodule for a ``SampleAdaptiveDropout``."""
        count = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Dropout):
                setattr(module, name, SampleAdaptiveDropout(child.p, state))
                count += 1
            else:
                count += Model._replace_dropouts(child, state)
        return count

    def dropout_rates(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (8)-(9): batch min-max scores, ``tanh(s * Softplus(gamma))``, rescale."""
        score = self.scorer(x)
        span = score.max() - score.min()
        if span > 1e-6:
            relative = (score - score.min()) / span
        else:
            relative = torch.full_like(score, 0.5)
        curve = torch.tanh(relative * F.softplus(self.sensitivity))
        return self.p_min + (self.p_max - self.p_min) * curve

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must have shape (batch, {self.seq_len}, {self.enc_in}), "
                f"got {tuple(x_enc.shape)}"
            )
        if not self.training:
            return self.backbone(x_enc)
        self.rate_state.rates = self.dropout_rates(x_enc)
        try:
            return self.backbone(x_enc)
        finally:
            self.rate_state.rates = None
