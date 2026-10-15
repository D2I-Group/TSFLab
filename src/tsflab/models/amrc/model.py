"""Clean-room AMRC training objective over a compact local forecaster.

AMRC is an optimization method, not a forecasting backbone. The point-forecast
``forward`` path therefore uses a small channel-independent encoder/predictor,
while :meth:`training_loss` exposes the paper's Adaptive Masking Loss (AML)
and Embedding Similarity Penalty (ESP). No reference implementation was used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        mask_samples: int = 4,
        lambda_aml: float = 0.1,
        lambda_esp: float = 0.1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.mask_samples = min(mask_samples, seq_len)
        self.lambda_aml = float(lambda_aml)
        self.lambda_esp = float(lambda_esp)
        self.use_revin = use_revin

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.predictor = nn.Linear(d_model, pred_len)

    def _validate(self, x: torch.Tensor) -> None:
        check_window(x, self.seq_len, self.enc_in)

    @staticmethod
    def prefix_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mask the oldest ``lengths[b]`` values of every sample."""
        if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
            raise ValueError("mask lengths must contain one value per sample")
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)
        return x.masked_fill(positions < lengths.view(-1, 1, 1), 0.0)

    def _forecast_and_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(x)
        normalized = self.revin(x, "norm") if self.use_revin else x
        embedding = self.encoder(normalized.transpose(1, 2))
        forecast = self.predictor(embedding).transpose(1, 2)
        if self.use_revin:
            forecast = self.revin(forecast, "denorm")
        return forecast, embedding

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        return self._forecast_and_embedding(x_enc)[0]

    @staticmethod
    def embedding_similarity_penalty(embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Equation (11)--(13): match pairwise embedding/target geometry."""
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError("embeddings and targets must share a batch dimension")
        batch = embeddings.shape[0]
        flat_embedding = embeddings.reshape(batch, -1)
        flat_target = targets.reshape(batch, -1)
        delta_embedding = (flat_embedding[:, None] - flat_embedding[None, :]).square().mean(-1)
        delta_target = (flat_target[:, None] - flat_target[None, :]).square().mean(-1)
        return (delta_embedding - delta_target).abs().mean()

    def adaptive_masking_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        mask_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Equation (6)--(10): follow the best sampled prefix mask per item."""
        forecast, embedding = self._forecast_and_embedding(x)
        if target.shape != forecast.shape:
            raise ValueError("target must match the forecast shape")
        original_loss = (forecast - target).square().mean(dim=(1, 2))
        if mask_lengths is None:
            # Even spacing is deterministic; callers can supply randomly sampled
            # lengths to reproduce the paper's stochastic training procedure.
            candidates = torch.linspace(1, self.seq_len, self.mask_samples, device=x.device).round().to(torch.long)
        else:
            candidates = mask_lengths.to(device=x.device, dtype=torch.long).flatten()
            if candidates.numel() < 1:
                raise ValueError("at least one mask length is required")
        candidates = candidates.clamp(1, self.seq_len)

        candidate_losses: list[torch.Tensor] = []
        candidate_embeddings: list[torch.Tensor] = []
        for length in candidates:
            lengths = length.expand(x.shape[0])
            masked_forecast, masked_embedding = self._forecast_and_embedding(self.prefix_mask(x, lengths))
            candidate_losses.append((masked_forecast - target).square().mean(dim=(1, 2)))
            candidate_embeddings.append(masked_embedding)

        losses = torch.stack(candidate_losses, dim=0)
        embeddings = torch.stack(candidate_embeddings, dim=0)
        best_loss, best_index = losses.min(dim=0)
        gather_index = best_index.view(1, -1, 1, 1).expand(1, x.shape[0], self.enc_in, embedding.shape[-1])
        best_embedding = embeddings.gather(0, gather_index).squeeze(0)
        beta = ((original_loss - best_loss) / original_loss.clamp_min(1e-8)).clamp_min(0)
        distance = (embedding - best_embedding).square().mean(dim=(1, 2))
        return (beta * distance).mean()

    def training_objective(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        mask_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Equation (14): prediction loss plus AML and ESP regularizers."""
        forecast, embedding = self._forecast_and_embedding(x)
        prediction = F.mse_loss(forecast, target)
        aml = self.adaptive_masking_loss(x, target, mask_lengths)
        esp = self.embedding_similarity_penalty(embedding, target)
        total = prediction + self.lambda_aml * aml + self.lambda_esp * esp
        return forecast, total, {"prediction": prediction, "aml": aml, "esp": esp}
