"""Independent PULSE forecasting core derived from the paper equations.

The implementation follows the paper's Disentangle--Evolve--Simulate path:
phase-codebook anchors, residual-only normalization, a two-stage phase router,
and coordinate-consistent residual denormalization. No reference repository
source was inspected or copied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


class PhaseRouter(nn.Module):
    """Two-stage cross-attention implementing paper Equations (4)--(5)."""

    def __init__(self, channels: int, d_model: int, heads: int) -> None:
        super().__init__()
        self.history_projection = nn.Linear(channels, d_model)
        self.future_projection = nn.Linear(channels, d_model)
        self.history_queries_future = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.future_queries_history = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, channels),
        )

    def forward(
        self,
        historical_anchor: torch.Tensor,
        latent_future: torch.Tensor,
        resolution: int,
    ) -> torch.Tensor:
        history = F.interpolate(
            historical_anchor.transpose(1, 2), resolution, mode="linear", align_corners=False
        ).transpose(1, 2)
        future = F.interpolate(latent_future.transpose(1, 2), resolution, mode="linear", align_corners=False).transpose(
            1, 2
        )
        history_tokens = self.history_projection(history)
        future_tokens = self.future_projection(future)
        routed_history, _ = self.history_queries_future(
            history_tokens, future_tokens, future_tokens, need_weights=False
        )
        evolved, _ = self.future_queries_history(future_tokens, routed_history, routed_history, need_weights=False)
        return self.output(evolved)


class Model(nn.Module):
    """PULSE point forecaster for ``(batch, time, channel)`` tensors."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        phase_period: int = 24,
        phase_resolution: int = 8,
        router_heads: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.phase_period = phase_period
        self.phase_resolution = phase_resolution
        self.eps = eps

        self.phase_codebook = nn.Parameter(torch.empty(phase_period, enc_in))
        nn.init.normal_(self.phase_codebook, std=0.02)
        self.backbone = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.router = PhaseRouter(enc_in, d_model, router_heads)

    def phase_indices(self, length: int, *, future: bool, device: torch.device) -> torch.Tensor:
        """Return the circular paper Equation (1) phase lookup indices."""
        if future:
            positions = torch.arange(1, length + 1, device=device)
            return (self.seq_len - 1 + positions).remainder(self.phase_period)
        positions = torch.arange(self.seq_len, device=device)
        return positions.remainder(self.phase_period)

    def disentangle(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply phase anchoring and residual-only normalization (Eq. 2--3)."""
        index = self.phase_indices(self.seq_len, future=False, device=x.device)
        anchor = self.phase_codebook[index].unsqueeze(0).expand(x.shape[0], -1, -1)
        residual = x - anchor
        mean = residual.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(residual.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        normalized = (residual - mean) / scale + anchor
        return normalized, anchor, mean, scale

    @staticmethod
    def statistic_aware_mixup(
        normalized: torch.Tensor,
        residual_mean: torch.Tensor,
        residual_scale: torch.Tensor,
        permutation: torch.Tensor,
        mixing: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate inputs and residual statistics without scale collapse (Eq. 7)."""
        weight = torch.as_tensor(mixing, dtype=normalized.dtype, device=normalized.device)
        while weight.ndim < normalized.ndim:
            weight = weight.unsqueeze(-1)
        mixed = weight * normalized + (1 - weight) * normalized[permutation]
        mean = weight * residual_mean + (1 - weight) * residual_mean[permutation]
        scale = weight * residual_scale + (1 - weight) * residual_scale[permutation]
        return mixed, mean, scale

    @staticmethod
    def frequency_mae(forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the paper's frequency-domain L1 objective (Eq. 9)."""
        return (torch.fft.rfft(forecast, dim=1) - torch.fft.rfft(target, dim=1)).abs().sum()

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized, historical_anchor, residual_mean, residual_scale = self.disentangle(x_enc)
        latent_future = self.backbone(normalized.transpose(1, 2)).transpose(1, 2)
        routed = self.router(historical_anchor, latent_future, self.phase_resolution)
        routed = F.interpolate(routed.transpose(1, 2), self.pred_len, mode="linear", align_corners=False).transpose(
            1, 2
        )
        future_index = self.phase_indices(self.pred_len, future=True, device=x_enc.device)
        future_anchor = routed + self.phase_codebook[future_index].unsqueeze(0)
        normalized_residual = latent_future - future_anchor
        return normalized_residual * residual_scale + residual_mean + future_anchor


__all__ = ["Model", "PhaseRouter"]
