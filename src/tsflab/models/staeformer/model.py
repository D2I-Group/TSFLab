"""Local STAEformer implementation from paper and reference-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window


class AxisAttentionBlock(nn.Module):
    """Pre-normalized attention and feed-forward update along one axis."""

    def __init__(self, width: int, heads: int, feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.feed_forward(self.norm2(x))


class Model(nn.Module):
    """Alternating temporal/spatial attention with adaptive embeddings."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        steps_per_day: int = 24,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
    ) -> None:
        super().__init__()
        del adj_mx
        if num_nodes < 1:
            raise ValueError("STAEformer dimensions must be positive")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.value_embedding = nn.Linear(1, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = (
            nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim)) if spatial_embedding_dim else None
        )
        self.adaptive_embedding = nn.Parameter(torch.empty(seq_len, num_nodes, adaptive_embedding_dim))
        width = (
            input_embedding_dim + tod_embedding_dim + dow_embedding_dim + spatial_embedding_dim + adaptive_embedding_dim
        )
        self.temporal_layers = nn.ModuleList(
            AxisAttentionBlock(width, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)
        )
        self.spatial_layers = nn.ModuleList(
            AxisAttentionBlock(width, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)
        )
        self.use_mixed_proj = use_mixed_proj
        self.output = (
            nn.Linear(seq_len * width, pred_len)
            if use_mixed_proj
            else nn.Sequential(nn.Linear(width, 1), nn.Flatten(start_dim=1), nn.Linear(seq_len, pred_len))
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)
        if self.node_embedding is not None:
            nn.init.xavier_uniform_(self.node_embedding)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        check_window(x_enc, self.seq_len, self.num_nodes)
        data = to_spatiotemporal(x_enc, x_mark_enc)
        tod = (data[..., 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
        dow = (data[..., 2] * 7).long().clamp(0, 6)
        pieces = [self.value_embedding(data[..., :1]), self.tod_embedding(tod), self.dow_embedding(dow)]
        if self.node_embedding is not None:
            pieces.append(
                self.node_embedding.view(1, 1, self.num_nodes, -1).expand(x_enc.shape[0], self.seq_len, -1, -1)
            )
        pieces.append(self.adaptive_embedding.unsqueeze(0).expand(x_enc.shape[0], -1, -1, -1))
        hidden = torch.cat(pieces, dim=-1)
        for temporal, spatial in zip(self.temporal_layers, self.spatial_layers):
            width = hidden.shape[-1]
            hidden = (
                temporal(hidden.transpose(1, 2).reshape(-1, self.seq_len, width))
                .reshape(x_enc.shape[0], self.num_nodes, self.seq_len, width)
                .transpose(1, 2)
            )
            hidden = spatial(hidden.reshape(-1, self.num_nodes, width)).reshape(
                x_enc.shape[0], self.seq_len, self.num_nodes, width
            )
        if self.use_mixed_proj:
            return self.output(hidden.transpose(1, 2).flatten(2)).transpose(1, 2)
        projected = self.output(hidden.transpose(1, 2).reshape(-1, self.seq_len, hidden.shape[-1]))
        return projected.reshape(x_enc.shape[0], self.num_nodes, self.pred_len).transpose(1, 2)


__all__ = ["Model", "AxisAttentionBlock"]
