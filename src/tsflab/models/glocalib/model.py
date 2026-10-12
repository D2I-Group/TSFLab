"""Clean-room forecasting adaptation of Glocal-IB (Yang et al., 2025).

The paper defines Glocal-IB as an imputation training paradigm rather than a
forecasting architecture. This module implements its disclosed bottleneck and
global-alignment equations around a small forecasting decoder; it does not
claim to reproduce the paper's imputation experiments.

Equation map (paper section 3): Eq. (6)-(8) is the diagonal Gaussian encoder
and analytic KL regularizer; Eq. (12)-(13) aligns a corrupted-view projection
with the stop-gradient complete-view latent; Eq. (14) supplies ``aux_loss``.
The runner's forecasting loss is the local/task term.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN


class _VariationalSequenceEncoder(nn.Module):
    """Produce the diagonal-Gaussian parameters in paper Eq. (6)."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(channels, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.mean = nn.Linear(width, width)
        self.log_variance = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.features(x)
        return self.mean(hidden), self.log_variance(hidden).clamp(-10.0, 10.0)


class Model(nn.Module):
    """Forecasting model trained with the Glocal-IB latent regularizers."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        align_weight: float = 0.5,
        mask_ratio: float = 0.25,
        align_loss_type: str = "cos_align",
        kl_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) <= 0:
            raise ValueError("sequence and horizon sizes must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mask_ratio = float(mask_ratio)
        self.align_weight = float(align_weight)
        self.kl_weight = float(kl_weight)
        self.align_loss_type = align_loss_type
        self.normalization = RevIN(enc_in, affine=False)
        self.encoder = _VariationalSequenceEncoder(enc_in, d_model)
        self.projector = nn.Linear(d_model, d_model)
        self.temporal_decoder = nn.Linear(seq_len, pred_len)
        self.value_decoder = nn.Linear(d_model, enc_in)
        self.aux_loss: torch.Tensor | None = None

    def _sample(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mean
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        keep = torch.rand((*x.shape[:2], 1), device=x.device) >= self.mask_ratio
        return x * keep.to(x.dtype)

    def _alignment(self, projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach()
        if self.align_loss_type == "cos_align":
            return 1.0 - F.cosine_similarity(projected, target, dim=-1).mean()
        projected = F.normalize(projected, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = torch.matmul(projected, target.transpose(1, 2))
        labels = torch.arange(projected.shape[1], device=projected.device)
        return F.cross_entropy(logits.flatten(0, 1), labels.repeat(projected.shape[0]))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len:
            raise ValueError(f"expected [batch, {self.seq_len}, channels], got {tuple(x_enc.shape)}")
        normalized = self.normalization(x_enc, "norm")
        mean, log_variance = self.encoder(normalized)
        latent = self._sample(mean, log_variance)
        horizon_latent = self.temporal_decoder(latent.transpose(1, 2)).transpose(1, 2)
        forecast = self.value_decoder(horizon_latent)
        forecast = self.normalization(forecast, "denorm")

        self.aux_loss = None
        if self.training:
            corrupt_mean, _ = self.encoder(self._corrupt(normalized))
            alignment = self._alignment(self.projector(corrupt_mean), mean)
            # D_KL(N(mu, diag(exp(logvar))) || N(0, I)), paper Eq. (8).
            kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).mean()
            self.aux_loss = self.kl_weight * kl + self.align_weight * alignment
        return forecast
