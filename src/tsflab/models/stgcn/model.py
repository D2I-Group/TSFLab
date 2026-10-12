"""Local STGCN implementation from the paper and reference-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.graph_spectral import chebyshev_supports
from tsflab.models._components.marks import to_spatiotemporal


class TemporalGate(nn.Module):
    """Causal temporal GLU preserving the input length."""

    def __init__(self, in_dim: int, out_dim: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(in_dim, 2 * out_dim, (kernel, 1))
        self.residual = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(F.pad(x, (0, 0, self.kernel - 1, 0)))
        value, gate = convolved.chunk(2, dim=1)
        return (value + self.residual(x)) * torch.sigmoid(gate)


class ChebyshevGraphConvolution(nn.Module):
    """STGCN spectral graph filter over fixed Chebyshev supports."""

    def __init__(self, in_dim: int, out_dim: int, supports: torch.Tensor, bias: bool) -> None:
        super().__init__()
        self.register_buffer("supports", supports)
        self.weight = nn.Parameter(torch.empty(supports.shape[0], in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neighborhoods = torch.einsum("knm,bctm->bktnc", self.supports, x)
        result = torch.einsum("bktnc,kco->botn", neighborhoods, self.weight)
        return result if self.bias is None else result + self.bias.view(1, -1, 1, 1)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, bottleneck: int, kernel: int, supports: torch.Tensor, bias: bool, dropout: float) -> None:
        super().__init__()
        self.temporal_in = TemporalGate(in_dim, hidden, kernel)
        self.graph = ChebyshevGraphConvolution(hidden, bottleneck, supports, bias)
        self.temporal_out = TemporalGate(bottleneck, hidden, kernel)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_in(x)
        x = torch.relu(self.graph(x))
        x = self.temporal_out(x)
        return self.dropout(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class Model(nn.Module):
    """Temporal-gated, Chebyshev-spatial, temporal-gated forecaster."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, input_dim: int = 3, Kt: int = 3, Ks: int = 3, hidden_dim: int = 64, bottleneck_dim: int = 16, out_hidden_dim: int = 128, act_func: str = "glu", graph_conv_type: str = "cheb_graph_conv", bias: bool = True, droprate: float = 0.5) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("STGCN dimensions must be positive")
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adj_mx shape must match num_nodes")
        supports = chebyshev_supports(adjacency, Ks)
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.block1 = SpatioTemporalBlock(input_dim, hidden_dim, bottleneck_dim, Kt, supports, bias, droprate)
        self.block2 = SpatioTemporalBlock(hidden_dim, hidden_dim, bottleneck_dim, Kt, supports, bias, droprate)
        self.forecast = nn.Sequential(nn.Linear(seq_len * hidden_dim, out_hidden_dim), nn.ReLU(), nn.Linear(out_hidden_dim, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"STGCN expects (B, {self.seq_len}, {self.num_nodes}) values")
        data = to_spatiotemporal(x_enc, x_mark_enc)
        if data.shape[-1] < self.input_dim:
            raise ValueError("STGCN received fewer input features than configured")
        hidden = data[..., : self.input_dim].permute(0, 3, 1, 2)
        hidden = self.block2(self.block1(hidden))
        return self.forecast(hidden.permute(0, 3, 2, 1).flatten(2)).transpose(1, 2)


__all__ = ["Model", "TemporalGate", "ChebyshevGraphConvolution", "SpatioTemporalBlock"]
