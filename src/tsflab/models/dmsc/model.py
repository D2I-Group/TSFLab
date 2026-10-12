"""Local DMSC (Dynamic Multi-Scale Coordination) implementation.

Independent rewrite of Section 3 (Eqs. 1-23) of Yang et al., arXiv 2508.02753,
with omissions resolved by reading the official repository
(``1327679995/DMSC``, revision ``57d03f1``). Nothing was copied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._components.topk_expert_router import GatingMLP

# EMPD projects every patch, whatever its length, to this many average-pooled
# samples before the linear projector (official ``AdaptiveAvgPool1d(8)``).
PATCH_POOL = 8


def base_patch_length(alpha: float, patch_min: int, patch_max: int) -> int:
    """Eq. (7): ``P_base = floor(P_min + alpha (P_max - P_min))`` clipped to the bounds."""
    raw = int(patch_min + (patch_max - patch_min) * alpha)
    return max(patch_min, min(patch_max, raw))


def layer_patch_length(base: int, layer: int, patch_min: int, decay: int) -> int:
    """Eq. (8): ``P_l = max(P_min, floor(P_base / tau^l))`` with ``l = 0`` for the first layer."""
    return max(patch_min, base // decay**layer)


def unfold_patches(x: torch.Tensor, patch_len: int) -> torch.Tensor:
    """Eq. (9): replicate-pad the end to a stride multiple, then unfold.

    ``[B, C, L] -> [B, C, N, P]`` with stride ``S = max(1, P // 2)``.
    """
    stride = max(1, patch_len // 2)
    pad = (stride - x.shape[-1] % stride) % stride
    if pad:
        x = F.pad(x, (0, pad), mode="replicate")
    return x.unfold(-1, patch_len, stride)


class ScaleFactorNet(nn.Module):
    """Eq. (6): ``alpha = N_theta(GAP(X))``, the input-adaptive patch scale in ``[0, 1]``.

    The variate-averaged series passes a 3-tap convolution and global average
    pooling, then an MLP with a sigmoid output; the batch mean is the scale.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.gelu(self.conv(x.mean(dim=1, keepdim=True))).mean(dim=-1)  # [B, 4]
        return self.dropout(self.mlp(pooled)).mean()


class PatchEmbedding(nn.Module):
    """Eq. (10): patches ``[B, C, N, P]`` -> ``[B, C, N, D]`` (pool to 8 samples, linear)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(PATCH_POOL)
        self.project = nn.Linear(PATCH_POOL, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, channels, count, length = patches.shape
        pooled = self.pool(patches.reshape(batch * channels, count, length))
        return self.dropout(self.project(pooled).reshape(batch, channels, count, -1))


def _separable_conv(d_model: int, dilation: int, dropout: float) -> list[nn.Module]:
    return [
        nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation, groups=d_model),
        nn.BatchNorm1d(d_model),
        nn.GELU(),
        nn.Conv1d(d_model, d_model, 1),
        nn.BatchNorm1d(d_model),
        nn.Dropout(dropout),
        nn.GELU(),
    ]


class TriadInteractionBlock(nn.Module):
    """TIB (Eqs. 11-16): intra-patch, inter-patch, and cross-variable dependencies."""

    def __init__(self, enc_in: int, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.intra = nn.Sequential(*_separable_conv(d_model, 1, dropout))  # Eq. (11)
        self.inter = nn.Sequential(*_separable_conv(d_model, 2, dropout), nn.AdaptiveAvgPool1d(1))  # Eq. (12)
        # Eq. (13): per-variate sigmoid gate from the variate-averaged inter-patch feature.
        self.cross = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, enc_in), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.fusion_gate = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.residual = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, C, N, D]`` patch embeddings -> ``[B, C, D]`` scale feature ``F_l``."""
        batch, channels, count, width = z.shape
        intra = self.intra(z.permute(0, 1, 3, 2).reshape(batch * channels, width, count))
        inter = self.inter(intra).squeeze(-1).reshape(batch, channels, width)
        gate = self.cross(inter.mean(dim=1, keepdim=True)).transpose(1, 2)  # [B, C, 1]
        cross = inter * gate  # Eq. (14)
        intra = intra.mean(dim=-1).reshape(batch, channels, width)
        weights = torch.softmax(self.fusion_gate(torch.cat([intra, inter, cross], dim=-1)), dim=-1)
        fused = weights[..., 0:1] * intra + weights[..., 1:2] * inter + weights[..., 2:3] * cross  # Eq. (15)
        out = self.norm(fused + self.residual(z.mean(dim=2)))  # Eq. (16)
        return self.dropout(out)


class AdaptiveScaleRoutingMoE(nn.Module):
    """ASR-MoE (Eqs. 17-22): global/local experts per scale, temporal-aware scale fusion."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        pred_len: int,
        num_scales: int,
        num_global: int,
        num_local: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_global, self.top_k = num_global, top_k
        # Global experts: depth 3; local experts: depth 2 (Table 2).
        self.global_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_ff), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_ff, pred_len),
            )
            for _ in range(num_global)
        )
        self.local_experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, pred_len))
            for _ in range(num_local)
        )
        self.router = GatingMLP(d_model, num_global + num_local, 2 * d_model, noisy=False)
        # Temporal-aware weighting (Eq. 20): descriptors of each scale forecast plus w_hist.
        self.descriptor = nn.Sequential(nn.Linear(pred_len, pred_len), nn.GELU(), nn.Dropout(dropout))
        self.history = nn.Parameter(torch.zeros(1, num_scales))
        self.scale_weight = nn.Sequential(nn.Linear(num_scales * pred_len + num_scales, num_scales), nn.Tanh(), nn.Softplus())
        self.output = nn.Sequential(
            nn.Linear(pred_len, pred_len), nn.GELU(), nn.Linear(pred_len, pred_len), nn.Dropout(dropout)
        )

    def scale_forecast(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (17)-(19), (21) inner sum: ``[B, C, D]`` -> forecast ``[B, C, H]`` and router weights."""
        batch, channels, width = feature.shape
        flat = feature.reshape(batch * channels, width)
        weights = self.router(flat)  # one softmax over global and local experts
        global_weights = weights[:, : self.num_global]
        local_weights, local_index = weights[:, self.num_global :].topk(self.top_k, dim=-1)
        global_out = torch.stack([expert(flat) for expert in self.global_experts], dim=1)
        local_out = torch.stack([expert(flat) for expert in self.local_experts], dim=1)
        chosen = local_out.gather(1, local_index.unsqueeze(-1).expand(-1, -1, local_out.shape[-1]))
        forecast = (global_out * global_weights.unsqueeze(-1)).sum(1) + (chosen * local_weights.unsqueeze(-1)).sum(1)
        return forecast.reshape(batch, channels, -1), weights

    def scale_weights(self, forecasts: list[torch.Tensor]) -> torch.Tensor:
        """Eq. (20): ``w = softmax(MLP(concat_l phi(Y_l) ++ w_hist))``, shape ``[B, L]``."""
        descriptors = [self.descriptor(forecast.mean(dim=1)) for forecast in forecasts]
        history = self.history.expand(forecasts[0].shape[0], -1)
        return torch.softmax(self.scale_weight(torch.cat(descriptors + [history], dim=-1)), dim=-1)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        forecasts, routing = zip(*(self.scale_forecast(feature) for feature in features))
        weights = self.scale_weights(list(forecasts))
        fused = sum(weights[:, index, None, None] * forecast for index, forecast in enumerate(forecasts))
        return self.output(fused), torch.cat(routing, dim=0)


def balance_loss(routing: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Eq. (22): ``-lambda * E[H(Omega)]``; minimizing it maximizes routing entropy."""
    entropy = -(routing * torch.log(routing + 1e-8)).sum(dim=-1).mean()
    return -coefficient * entropy


class Model(nn.Module):
    """DMSC: EMPD-TIB progressive cascade with an ASR-MoE prediction head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        e_layers: int = 3,
        d_model: int = 128,
        d_ff: int = 256,
        patch_min: int = 8,
        patch_max: int = 64,
        patch_decay: int = 2,
        num_global: int = 2,
        num_local: int = 8,
        top_k: int = 2,
        balance_coeff: float = 0.2,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if patch_max > seq_len:
            raise ValueError("DMSC needs 1 <= patch_min <= patch_max <= seq_len")
        self.seq_len, self.pred_len, self.enc_in, self.e_layers = seq_len, pred_len, enc_in, e_layers
        self.patch_min, self.patch_max, self.patch_decay = patch_min, patch_max, patch_decay
        self.balance_coeff = balance_coeff
        self.revin = RevIN(enc_in, affine=True, enabled=use_norm)
        self.scale_factor = ScaleFactorNet(dropout)
        self.embeddings = nn.ModuleList(PatchEmbedding(d_model, dropout) for _ in range(e_layers))
        self.tibs = nn.ModuleList(TriadInteractionBlock(enc_in, d_model, d_ff, dropout) for _ in range(e_layers))
        # Gated pathway P_l of Eq. (3): a sigmoid gate times a projection back to the input length.
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())
            for _ in range(e_layers - 1)
        )
        self.feedback = nn.ModuleList(nn.Linear(d_model, seq_len) for _ in range(e_layers - 1))
        self.moe = AdaptiveScaleRoutingMoE(d_model, d_ff, pred_len, e_layers, num_global, num_local, top_k, dropout)
        self.linear_residual = nn.Sequential(nn.Linear(seq_len, d_ff), nn.GELU(), nn.Linear(d_ff, pred_len))
        self.last_balance_loss: torch.Tensor | None = None

    def patch_lengths(self, x: torch.Tensor) -> list[int]:
        """Eqs. (6)-(8): one data-dependent base length per batch, decayed per layer."""
        alpha = float(self.scale_factor(x).detach())
        base = base_patch_length(alpha, self.patch_min, self.patch_max)
        return [layer_patch_length(base, layer, self.patch_min, self.patch_decay) for layer in range(self.e_layers)]

    def cascade(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Eqs. (1)-(5): ``[B, C, L]`` -> scale features ``[F_1, ..., F_L]`` each ``[B, C, D]``."""
        features: list[torch.Tensor] = []
        for layer, patch_len in enumerate(self.patch_lengths(x)):
            if features:
                previous = features[-1]
                x_in = x + self.gates[layer - 1](previous) * self.feedback[layer - 1](previous)
            else:
                x_in = x
            z = self.embeddings[layer](unfold_patches(x_in, patch_len))
            features.append(self.tibs[layer](z))
        return features

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        x = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        forecast, routing = self.moe(self.cascade(x))
        self.last_balance_loss = balance_loss(routing, self.balance_coeff)
        out = forecast + self.linear_residual(x)
        return self.revin(out.transpose(1, 2), "denorm")


__all__ = [
    "AdaptiveScaleRoutingMoE",
    "Model",
    "PatchEmbedding",
    "ScaleFactorNet",
    "TriadInteractionBlock",
    "balance_loss",
    "base_patch_length",
    "layer_patch_length",
    "unfold_patches",
]
