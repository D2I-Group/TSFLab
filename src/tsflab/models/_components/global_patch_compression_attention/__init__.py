"""Two-stage cross-patch attention with global-patch compression.

Pure cross-patch self-attention over every group's patches (for example every
variable's temporal patches) costs ``O(groups^2 * patches^2 * d_model)``. This
operator instead compresses each group's patches into a single summary token
via cross-attention (stage 1), then lets every patch attend back to the pooled
summaries (stage 2), reducing cost to ``O(groups^2 * patches * d_model)`` while
still letting any patch of any group influence any other patch through the
shared summaries. It is a paper-neutral "compress tokens, then attend through
the compressed set" primitive usable by any patch-token forecaster that wants
global cross-group context without quadratic patch-count cost.
"""

from __future__ import annotations

import torch
from torch import nn


class GlobalPatchCompressionAttention(nn.Module):
    """Compress each group's patches into one summary, then attend through it."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_ff < 1:
            raise ValueError("d_model, n_heads, and d_ff must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.compress_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.broadcast_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_compress_attn = nn.LayerNorm(d_model)
        self.norm_compress_mlp = nn.LayerNorm(d_model)
        self.norm_broadcast_attn = nn.LayerNorm(d_model)
        self.norm_broadcast_mlp = nn.LayerNorm(d_model)
        self.compress_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.broadcast_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Refine ``patches`` shaped ``[batch, groups, patches_per_group, d_model]``."""
        if patches.ndim != 4:
            raise ValueError("patches must have shape [batch, groups, patches_per_group, d_model]")
        batch, groups, per_group, d_model = patches.shape
        flat = patches.reshape(batch, groups * per_group, d_model)

        query = patches[:, :, -1, :]
        compressed, _ = self.compress_attn(query, flat, flat)
        sensor = self.norm_compress_attn(query + compressed)
        sensor = self.norm_compress_mlp(sensor + self.compress_mlp(sensor))

        broadcast, _ = self.broadcast_attn(flat, sensor, sensor)
        refined = self.norm_broadcast_attn(flat + broadcast)
        refined = self.norm_broadcast_mlp(refined + self.broadcast_mlp(refined))
        return refined.reshape(batch, groups, per_group, d_model)


__all__ = ["GlobalPatchCompressionAttention"]
