"""Clean-room Spatial-Temporal Transformer Network (arXiv:2001.02908)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


def _row_normalize(adjacency: np.ndarray, nodes: int) -> torch.Tensor:
    matrix = square_adjacency(adjacency, nodes) + np.eye(nodes, dtype=np.float32)
    return torch.from_numpy(matrix / np.maximum(matrix.sum(-1, keepdims=True), 1e-8))


class SpatialTransformer(nn.Module):
    """Dynamic directed attention fused with stationary graph convolution."""

    def __init__(self, width: int, heads: int, adjacency: torch.Tensor, expand: int, dropout: float) -> None:
        super().__init__()
        self.dynamic = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.fixed = nn.Linear(width, width)
        self.gate = nn.Linear(2 * width, width)
        self.ffn = nn.Sequential(
            nn.Linear(width, expand * width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(expand * width, width)
        )
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.register_buffer("adjacency", adjacency)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        flat = x.reshape(b * t, n, d)
        dynamic, weights = self.dynamic(flat, flat, flat, need_weights=True, average_attn_weights=False)
        fixed = self.fixed(torch.einsum("nm,btmd->btnd", self.adjacency, x).reshape(b * t, n, d))
        gate = torch.sigmoid(self.gate(torch.cat((dynamic, fixed), -1)))
        fused = gate * dynamic + (1 - gate) * fixed
        fused = self.norm1(flat + fused)
        self.last_attention = weights.reshape(b, t, weights.shape[1], n, n)
        return self.norm2(fused + self.ffn(fused)).reshape(b, t, n, d)


class TemporalTransformer(nn.Module):
    """Bidirectional long-range attention along time for every sensor."""

    def __init__(self, width: int, heads: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(width, expand * width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(expand * width, width)
        )
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        flat = x.transpose(1, 2).reshape(b * n, t, d)
        attended = self.attention(flat, flat, flat, need_weights=False)[0]
        flat = self.norm1(flat + attended)
        return self.norm2(flat + self.ffn(flat)).reshape(b, n, t, d).transpose(1, 2)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, width: int, heads: int, adjacency: torch.Tensor, expand: int, dropout: float) -> None:
        super().__init__()
        self.spatial = SpatialTransformer(width, heads, adjacency, expand, dropout)
        self.temporal = TemporalTransformer(width, heads, expand, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(x))


class Model(nn.Module):
    """Stacked spatial→temporal blocks and direct multi-horizon head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        d_model: int = 64,
        mlp_expand: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in, self.cov_dim = seq_len, pred_len, enc_in, cov_dim
        if adj_mx is None:
            adj_mx = np.ones((enc_in, enc_in), dtype=np.float32)
        adjacency = _row_normalize(adj_mx, enc_in)
        self.embedding = nn.Linear(1 + cov_dim, d_model)
        self.position = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            SpatialTemporalBlock(d_model, 4, adjacency, mlp_expand, dropout) for _ in range(num_layers)
        )
        self.forecast = nn.Linear(seq_len * d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        inputs = to_spatiotemporal(x_enc, x_mark_enc)[..., : 1 + self.cov_dim]
        if inputs.shape[-1] < 1 + self.cov_dim:
            inputs = torch.nn.functional.pad(inputs, (0, 1 + self.cov_dim - inputs.shape[-1]))
        hidden = self.embedding(inputs) + self.position[None, :, None]
        for block in self.blocks:
            hidden = block(hidden)
        return self.forecast(hidden.transpose(1, 2).flatten(2)).transpose(1, 2)
