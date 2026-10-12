"""Local Extralonger implementation from paper and reference-code review.

Extralonger (arXiv:2411.00844, NeurIPS 2024 Workshop) targets *extra-long-term*
traffic forecasting (0.5 day to 1 week ahead) by computing three parallel
"routes" over the same time/value/calendar input and fusing them with a fixed
weighting instead of picking temporal-only or spatial-only processing:

* A **temporal route**: the whole node axis is linearly compressed into a
  per-time-step embedding (``t_input``), time-of-day/day-of-week embeddings are
  concatenated, and a stack of temporal self-attention blocks mixes across the
  time axis; the result is projected back out to ``(pred_len, num_nodes)``.
* A **spatial route**: the whole time axis is linearly compressed into a
  per-node embedding (``s_input``), a learned node identity embedding is
  concatenated, and a stack of *global-local* graph-attention blocks mixes
  across the node axis, blending dense (global) attention with an
  adjacency-masked (local) attention; the result is projected to
  ``(pred_len, num_nodes)``.
* A **mixed route**: the same temporal-route features are first passed
  through temporal self-attention, transposed so the (small) feature axis
  becomes the token axis and the time axis becomes the per-token width, run
  through a second self-attention stack over that transposed layout, and
  projected to ``(pred_len, num_nodes)``.

The three route outputs are fused with the official hand-set weights
``(temporal + spatial + 2 * mixed) / 4``.

The official global-local graph-attention equation (paper: ``GLSAtt =
(Softmax(alpha_local) + Softmax(alpha_global)) @ V / 2``) is paper-neutral and
reusable by other graph-attention forecasters, so it is imported from
``tsflab.models._components.graph_masked_attention`` instead of being re-derived
here. The route layout, compression/expansion projections, and fixed fusion
weights are paper-specific glue and stay model-local.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.graph_masked_attention import GlobalLocalGraphAttention
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window


class TransposeAxes(nn.Module):
    """Swap two tensor axes; used to move between time-major and node-major layouts."""

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


class SelfAttentionBlock(nn.Module):
    """Post-norm residual attention + feed-forward block over a token axis.

    Matches the official ``SelfAttentionLayer``: attention residual is added
    then layer-normalized, followed by a ReLU feed-forward residual that is
    added then layer-normalized. The attention itself is
    :class:`~models._components.graph_masked_attention.GlobalLocalGraphAttention`,
    used with ``adj_mask=None`` for the temporal/mixed routes (plain dense
    self-attention) and with a real adjacency mask for the spatial route.
    """

    def __init__(self, model_dim: int, feed_forward_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = GlobalLocalGraphAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        attended = self.attention(x, x, x, adj_mask=adj_mask)
        x = self.norm1(residual + self.dropout1(attended))
        residual = x
        return self.norm2(residual + self.dropout2(self.feed_forward(x)))


def _adjacency_mask(adj_mx: np.ndarray | None, num_nodes: int) -> torch.Tensor:
    """Boolean attention mask: the graph's nonzero entries plus self-loops (identity without a graph)."""
    if adj_mx is None:
        return torch.eye(num_nodes, dtype=torch.bool)
    dense = torch.as_tensor(np.asarray(adj_mx), dtype=torch.float32)
    return (dense != 0) | torch.eye(num_nodes, dtype=torch.bool)


def _attention_blocks(width: int, feed_forward_dim: int, num_heads: int, dropout: float, num_layers: int):
    return nn.ModuleList(SelfAttentionBlock(width, feed_forward_dim, num_heads, dropout) for _ in range(num_layers))


class Model(nn.Module):
    """Three-route (temporal / spatial / mixed) fused graph forecaster."""

    _MIX_SPATIAL_HEADS = 2  # matches the official fixed head count for the mixed route's node-axis pass.

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        steps_per_day: int = 288,
        input_embedding_dim: int = 8,
        tod_embedding_dim: int = 4,
        dow_embedding_dim: int = 4,
        spatial_embedding_dim: int = 8,
        feed_forward_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("Extralonger dimensions must be positive")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim

        self.register_buffer("adj_mask", _adjacency_mask(adj_mx, num_nodes))

        # temporal/mixed model width: value + tod + dow embeddings (no node embedding).
        self.model_dim = input_embedding_dim + tod_embedding_dim + dow_embedding_dim
        self._check_widths()

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
        nn.init.xavier_uniform_(self.node_embedding)

        # temporal route: compress the node axis, then attend over time.
        self.t_input = nn.Linear(num_nodes, input_embedding_dim)
        self.temporal_layers = _attention_blocks(self.model_dim, feed_forward_dim, num_heads, dropout, num_layers)
        self.t_output = self._temporal_output()

        # spatial route: compress the time axis, then attend over nodes with the adjacency blend.
        self.s_input = nn.Linear(seq_len, self.model_dim - spatial_embedding_dim)
        self.spatial_layers = _attention_blocks(self.model_dim, feed_forward_dim, num_heads, dropout, num_layers)
        self.s_output = nn.Linear(self.model_dim, pred_len)

        # mixed route: temporal attention, then a second attention pass with the feature axis as tokens.
        self.mix_temporal_layers = _attention_blocks(self.model_dim, feed_forward_dim, num_heads, dropout, num_layers)
        self.mix_spatial_layers = _attention_blocks(
            seq_len, feed_forward_dim, self._MIX_SPATIAL_HEADS, dropout, num_layers
        )
        self.mix_proj = nn.Sequential(
            TransposeAxes(1, 2),
            nn.Linear(self.model_dim, num_nodes),
            TransposeAxes(1, 2),
            nn.Linear(seq_len, pred_len),
            TransposeAxes(1, 2),
        )

    def _check_widths(self) -> None:
        if self.seq_len % self._MIX_SPATIAL_HEADS:
            raise ValueError("Extralonger seq_len must be divisible by the mixed route's fixed head count (2)")

    def _temporal_output(self) -> nn.Module:
        if self.seq_len != self.pred_len:
            return nn.Sequential(
                nn.Linear(self.model_dim, self.num_nodes),
                TransposeAxes(1, 2),
                nn.Linear(self.seq_len, self.pred_len),
                TransposeAxes(1, 2),
            )
        return nn.Linear(self.model_dim, self.num_nodes)

    def _temporal_tokens(self, history: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pieces = [self.t_input(values)]  # (B, T, input_embedding_dim)
        if self.tod_embedding_dim > 0:
            tod = (history[:, :, 0, 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            pieces.append(self.tod_embedding(tod))
        if self.dow_embedding_dim > 0:
            dow = (history[:, :, 0, 2] * 7).long().clamp(0, 6)
            pieces.append(self.dow_embedding(dow))
        return torch.cat(pieces, dim=-1)  # (B, T, model_dim)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.num_nodes)
        batch_size = x_enc.shape[0]
        history = to_spatiotemporal(x_enc, x_mark_enc)
        values = history[..., 0]  # (B, T, N)

        x_t = self._temporal_tokens(history, values)  # (B, T, model_dim)
        x_mix = x_t

        for layer in self.temporal_layers:
            x_t = layer(x_t)
        x_t = self.t_output(x_t)  # (B, pred_len, N)

        for layer in self.mix_temporal_layers:
            x_mix = layer(x_mix)
        x_mix = x_mix.transpose(1, 2)  # (B, model_dim, T)
        for layer in self.mix_spatial_layers:
            x_mix = layer(x_mix)
        x_mix = self.mix_proj(x_mix)  # (B, pred_len, N)

        x_s = self.s_input(values.transpose(1, 2))  # (B, N, model_dim - spatial_embedding_dim)
        x_s = torch.cat([x_s, self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
        for layer in self.spatial_layers:
            x_s = layer(x_s, adj_mask=self.adj_mask)
        x_s = self.s_output(x_s).transpose(1, 2)  # (B, pred_len, N)

        return (2.0 * x_mix + x_s + x_t) / 4.0


__all__ = ["Model", "SelfAttentionBlock", "TransposeAxes"]
