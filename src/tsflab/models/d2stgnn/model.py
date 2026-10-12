"""Local D2STGNN implementation from paper and official-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


def _propagate(x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    if graph.ndim == 2:
        return torch.einsum("blnc,nm->blmc", x, graph)
    return torch.einsum("blnc,bnm->blmc", x, graph)


class DynamicGraphConstructor(nn.Module):
    """Construct the paper's hidden-state-dependent directed graph."""

    def __init__(self, hidden: int, node_dim: int, nodes: int) -> None:
        super().__init__()
        self.node_source = nn.Parameter(torch.empty(nodes, node_dim))
        self.node_target = nn.Parameter(torch.empty(nodes, node_dim))
        self.query = nn.Linear(hidden + node_dim, node_dim)
        self.key = nn.Linear(hidden + node_dim, node_dim)
        nn.init.xavier_uniform_(self.node_source)
        nn.init.xavier_uniform_(self.node_target)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        summary = hidden.mean(dim=1)
        source = torch.cat((summary, self.node_source.unsqueeze(0).expand(summary.shape[0], -1, -1)), -1)
        target = torch.cat((summary, self.node_target.unsqueeze(0).expand(summary.shape[0], -1, -1)), -1)
        return torch.softmax(self.query(source) @ self.key(target).transpose(-1, -2) * self.query.out_features**-0.5, dim=-1)


class DecoupledLayer(nn.Module):
    """Separate diffusion and inherent signals, then subtract their backcasts."""

    def __init__(self, hidden: int, graphs: int, spatial_order: int, temporal_kernel: int, dropout: float, *, with_backcast: bool) -> None:
        super().__init__()
        self.spatial_order = spatial_order
        self.diffusion_projection = nn.Linear(hidden * (1 + graphs * spatial_order), hidden)
        self.inherent = nn.GRU(hidden, hidden, batch_first=True)
        self.temporal = nn.Conv1d(hidden, hidden, temporal_kernel, padding=temporal_kernel // 2)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.backcast = nn.Linear(2 * hidden, hidden) if with_backcast else None
        self.forecast = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: torch.Tensor, graphs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        terms = [residual]
        for graph in graphs:
            value = residual
            for _ in range(self.spatial_order):
                value = _propagate(value, graph)
                terms.append(value)
        diffusion = torch.tanh(self.diffusion_projection(torch.cat(terms, -1)))
        batch, length, nodes, hidden = residual.shape
        inherent, _ = self.inherent(residual.transpose(1, 2).reshape(batch * nodes, length, hidden))
        inherent = self.temporal(inherent.transpose(1, 2)).transpose(1, 2).reshape(batch, nodes, length, hidden).transpose(1, 2)
        weight = torch.sigmoid(self.gate(torch.cat((diffusion, inherent), -1)))
        separated = torch.cat((weight * diffusion, (1 - weight) * inherent), -1)
        next_residual = residual
        if self.backcast is not None:
            next_residual = residual - self.dropout(self.backcast(separated))
        return next_residual, self.forecast(separated[:, -1])


class Model(nn.Module):
    """Decoupled dynamic spatial-temporal graph neural forecaster."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, input_dim: int = 3, num_feat: int = 1, num_hidden: int = 16, node_hidden: int = 8, time_emb_dim: int = 8, k_s: int = 2, k_t: int = 3, gap: int = 1, num_layers: int = 2, dropout: float = 0.1, time_in_day_size: int = 288, day_in_week_size: int = 7, forecast_dim: int = 64, output_hidden: int = 128) -> None:
        super().__init__()
        del num_feat, output_hidden
        if pred_len % gap:
            raise ValueError("D2STGNN pred_len must be divisible by gap")
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.time_in_day_size, self.day_in_week_size = time_in_day_size, day_in_week_size
        self.tod_embedding = nn.Embedding(time_in_day_size, time_emb_dim)
        self.dow_embedding = nn.Embedding(day_in_week_size, time_emb_dim)
        self.input_projection = nn.Linear(input_dim + 2 * time_emb_dim, num_hidden)
        self.graph = DynamicGraphConstructor(num_hidden, node_hidden, num_nodes)
        self.adaptive_source = nn.Parameter(torch.empty(num_nodes, node_hidden))
        self.adaptive_target = nn.Parameter(torch.empty(node_hidden, num_nodes))
        self.layers = nn.ModuleList(
            DecoupledLayer(num_hidden, 4, k_s, k_t, dropout, with_backcast=layer < num_layers - 1)
            for layer in range(num_layers)
        )
        self.forecast = nn.Sequential(nn.Linear(num_layers * num_hidden, forecast_dim), nn.ReLU(), nn.Linear(forecast_dim, pred_len))
        nn.init.xavier_uniform_(self.adaptive_source)
        nn.init.xavier_uniform_(self.adaptive_target)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        check_window(x_enc, self.seq_len, self.num_nodes)
        data = to_spatiotemporal(x_enc, x_mark_enc)
        tod = (data[..., 1] * self.time_in_day_size).long().clamp(0, self.time_in_day_size - 1)
        dow = (data[..., 2] * self.day_in_week_size).long().clamp(0, self.day_in_week_size - 1)
        features = torch.cat((data[..., : self.input_dim], self.tod_embedding(tod), self.dow_embedding(dow)), -1)
        residual = self.input_projection(features)
        dynamic = self.graph(residual)
        adaptive = adaptive_node_embedding_adjacency(self.adaptive_source, self.adaptive_target)
        graphs = [self.forward_support, self.reverse_support, adaptive, dynamic]
        forecasts = []
        for layer in self.layers:
            residual, partial = layer(residual, graphs)
            forecasts.append(partial)
        combined = torch.cat(forecasts, -1)
        return self.forecast(combined).transpose(1, 2)


__all__ = ["Model", "DynamicGraphConstructor", "DecoupledLayer"]
