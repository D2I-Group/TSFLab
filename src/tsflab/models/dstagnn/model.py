"""Independent DSTAGNN implementation from the model's disclosed operations."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from tsflab.models._components.graph_spectral import chebyshev_supports


class AxisAttention(nn.Module):
    """Multi-head scaled dot-product attention over one explicit axis."""

    def __init__(self, model_dim: int, heads: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query = nn.Linear(model_dim, heads * key_dim)
        self.key = nn.Linear(model_dim, heads * key_dim)
        self.value = nn.Linear(model_dim, heads * value_dim)
        self.output = nn.Linear(heads * value_dim, model_dim)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        query = self.query(x).view(batch, length, self.heads, self.key_dim).transpose(1, 2)
        key = self.key(x).view(batch, length, self.heads, self.key_dim).transpose(1, 2)
        value = self.value(x).view(batch, length, self.heads, self.value_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.key_dim)
        if allowed is not None:
            scores = scores.masked_fill(~allowed[None, None], -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, -1)
        return self.output(mixed), weights.mean(1)


class DynamicChebyshevConvolution(nn.Module):
    """Chebyshev graph convolution modulated by per-sample spatial attention."""

    def __init__(self, supports: torch.Tensor, model_dim: int) -> None:
        super().__init__()
        self.register_buffer("supports", supports)
        self.weights = nn.Parameter(torch.empty(supports.shape[0], model_dim, model_dim))
        self.bias = nn.Parameter(torch.zeros(model_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        pieces = []
        for order, support in enumerate(self.supports):
            graph = support[None, None] * spatial_attention
            propagated = torch.einsum("btij,btjf->btif", graph, x)
            pieces.append(torch.einsum("btif,fo->btio", propagated, self.weights[order]))
        return torch.relu(torch.stack(pieces).sum(0) + self.bias)


class MultiScaleGatedTemporalConvolution(nn.Module):
    """Parallel gated temporal filters used by DSTAGNN's temporal module."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.filters = nn.ModuleList(
            nn.Conv2d(model_dim, model_dim, (1, kernel), padding=(0, kernel // 2))
            for kernel in (3, 5, 7)
        )
        self.gates = nn.ModuleList(
            nn.Conv2d(model_dim, model_dim, (1, kernel), padding=(0, kernel // 2))
            for kernel in (3, 5, 7)
        )
        self.fusion = nn.Conv2d(3 * model_dim, model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.permute(0, 3, 2, 1)
        branches = [
            torch.tanh(filter_layer(channels_first)) * torch.sigmoid(gate_layer(channels_first))
            for filter_layer, gate_layer in zip(self.filters, self.gates)
        ]
        return self.fusion(torch.cat(branches, dim=1)).permute(0, 3, 2, 1)


class DSTAGNNBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        nodes: int,
        model_dim: int,
        heads: int,
        key_dim: int,
        value_dim: int,
        supports: torch.Tensor,
        allowed: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("allowed_edges", allowed)
        self.temporal_attention = AxisAttention(model_dim, heads, key_dim, value_dim)
        self.spatial_attention = AxisAttention(model_dim, heads, key_dim, value_dim)
        self.graph_convolution = DynamicChebyshevConvolution(supports, model_dim)
        self.temporal_convolution = MultiScaleGatedTemporalConvolution(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)
        self.seq_len = seq_len
        self.nodes = nodes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, width = x.shape
        temporal_input = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        temporal, _ = self.temporal_attention(temporal_input)
        temporal = temporal.reshape(batch, nodes, steps, width).permute(0, 2, 1, 3)
        x = self.first_norm(x + temporal)

        spatial_input = x.reshape(batch * steps, nodes, width)
        spatial, attention = self.spatial_attention(spatial_input, self.allowed_edges)
        spatial = spatial.reshape(batch, steps, nodes, width)
        attention = attention.reshape(batch, steps, nodes, nodes)
        graph = self.graph_convolution(spatial, attention)
        return self.second_norm(x + graph + self.temporal_convolution(graph)), attention


class Model(nn.Module):
    """Dynamic spatial-temporal aware graph forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        d_model: int = 64,
        d_k: int = 8,
        d_v: int = 8,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths, nodes, and attention dimensions must be positive")
        adjacency = np.eye(enc_in, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        supports = chebyshev_supports(adjacency, 3)
        # Dynamic attention remains dense; the predefined graph enters the
        # Chebyshev filter rather than becoming a hard attention mask.
        allowed = torch.ones(enc_in, enc_in, dtype=torch.bool)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = enc_in
        self.value_embedding = nn.Linear(1, d_model)
        self.temporal_position = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.spatial_position = nn.Parameter(torch.randn(enc_in, d_model) * 0.02)
        self.block = DSTAGNNBlock(seq_len, enc_in, d_model, n_heads, d_k, d_v, supports, allowed)
        self.forecast = nn.Conv2d(seq_len, pred_len, kernel_size=(1, d_model))
        self.last_spatial_attention: torch.Tensor | None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        x = self.value_embedding(x_enc[..., None])
        x = x + self.temporal_position[None, :, None] + self.spatial_position[None, None]
        x, self.last_spatial_attention = self.block(x)
        return self.forecast(x).squeeze(-1)
