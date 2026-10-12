"""Local Graph WaveNet implementation from paper and reference-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.diffusion_conv import DiffusionConv2d
from tsflab.models._components.gated_dilated_conv import gated_dilated_conv
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal


class WaveNetGraphLayer(nn.Module):
    """Dilated gated temporal convolution followed by graph diffusion."""

    def __init__(self, channels: int, skip: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.filter = nn.Conv2d(channels, channels, (1, kernel), dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, kernel), dilation=(1, dilation))
        self.diffusion = DiffusionConv2d(channels, channels, dropout, support_len=3, order=2)
        self.residual = nn.Conv2d(channels, channels, 1)
        self.skip = nn.Conv2d(channels, skip, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        gated = gated_dilated_conv(x, self.filter, self.gate)
        mixed = self.diffusion(gated, supports)
        hidden = self.norm(self.residual(mixed) + x)
        return hidden, self.skip(hidden)


class Model(nn.Module):
    """Causal WaveNet backbone with predefined and learned graph supports."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, input_dim: int = 3, dropout: float = 0.3, residual_channels: int = 16, dilation_channels: int = 16, skip_channels: int = 64, end_channels: int = 128, kernel_size: int = 2, blocks: int = 2, layers: int = 2) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("GWNet dimensions must be positive")
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adj_mx shape must match num_nodes")
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        node_dim = min(10, max(2, num_nodes))
        self.source_nodes = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.target_nodes = nn.Parameter(torch.empty(node_dim, num_nodes))
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.input_projection = nn.Conv2d(input_dim, residual_channels, 1)
        self.layers = nn.ModuleList(
            WaveNetGraphLayer(residual_channels, skip_channels, kernel_size, 2**layer, dropout)
            for _ in range(blocks)
            for layer in range(layers)
        )
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(skip_channels, end_channels, 1), nn.ReLU(), nn.Conv2d(end_channels, pred_len, 1))
        nn.init.xavier_uniform_(self.source_nodes)
        nn.init.xavier_uniform_(self.target_nodes)

    def graph_supports(self) -> list[torch.Tensor]:
        adaptive = adaptive_node_embedding_adjacency(self.source_nodes, self.target_nodes)
        return [self.forward_support, self.reverse_support, adaptive]

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"GWNet expects (B, {self.seq_len}, {self.num_nodes}) values")
        data = to_spatiotemporal(x_enc, x_mark_enc)
        if data.shape[-1] < self.input_dim:
            raise ValueError("GWNet received fewer input features than configured")
        hidden = self.input_projection(data[..., : self.input_dim].permute(0, 3, 2, 1))
        skip_total = None
        supports = self.graph_supports()
        for layer in self.layers:
            hidden, skip = layer(hidden, supports)
            skip_total = skip if skip_total is None else skip_total + skip
        assert skip_total is not None
        return self.output(skip_total)[..., -1]


__all__ = ["Model", "WaveNetGraphLayer"]
