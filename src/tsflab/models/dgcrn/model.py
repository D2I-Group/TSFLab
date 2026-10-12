"""Paper-derived DGCRN with state-conditioned dynamic graph filters."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from tsflab.models._components.graph_conv_gru import GraphConvGRUCell
from tsflab.models._components.marks import coerce_time_length, to_spatiotemporal


def _transition(adjacency: np.ndarray) -> torch.Tensor:
    adjacency = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    denominator = np.maximum(adjacency.sum(1, keepdims=True), 1e-12)
    return torch.as_tensor(adjacency / denominator, dtype=torch.float32)


def _time_driver(values: torch.Tensor, marks: torch.Tensor | None, steps: int) -> torch.Tensor:
    """Return one known calendar driver per node without future target values."""
    if marks is None:
        return values.new_zeros((values.shape[0], steps, values.shape[2], 1))
    marks = coerce_time_length(marks, steps)
    structured = to_spatiotemporal(values.new_zeros((values.shape[0], steps, values.shape[2])), marks)
    if structured.shape[-1] == 1:
        return structured.new_zeros((*structured.shape[:-1], 1))
    return structured[..., 1:2]


class DynamicGraphGenerator(nn.Module):
    """Hyper-network that maps hidden state and node identity to directed graphs."""

    def __init__(self, nodes: int, hidden: int, node_dim: int, hyper_dim: int, middle_dim: int, alpha: float) -> None:
        super().__init__()
        self.source_embedding = nn.Parameter(torch.randn(nodes, node_dim) / math.sqrt(node_dim))
        self.target_embedding = nn.Parameter(torch.randn(nodes, node_dim) / math.sqrt(node_dim))
        self.source_hyper = nn.Sequential(
            nn.Linear(hidden + node_dim, hyper_dim), nn.Tanh(),
            nn.Linear(hyper_dim, middle_dim), nn.Tanh(), nn.Linear(middle_dim, node_dim),
        )
        self.target_hyper = nn.Sequential(
            nn.Linear(hidden + node_dim, hyper_dim), nn.Tanh(),
            nn.Linear(hyper_dim, middle_dim), nn.Tanh(), nn.Linear(middle_dim, node_dim),
        )
        self.alpha = alpha

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = hidden.shape[0]
        source = self.source_hyper(torch.cat((hidden, self.source_embedding.expand(batch, -1, -1)), dim=-1))
        target = self.target_hyper(torch.cat((hidden, self.target_embedding.expand(batch, -1, -1)), dim=-1))
        scores = torch.tanh(self.alpha * torch.matmul(source, target.transpose(-1, -2)))
        # Preserve signed state-dependent affinity before row normalization.
        # Applying ReLU here can collapse every negative row to the same uniform
        # graph, making the supposedly dynamic support locally insensitive to
        # hidden-state changes.
        forward = torch.softmax(scores, dim=-1)
        backward = torch.softmax(-scores.transpose(-1, -2), dim=-1)
        return forward, backward


class DynamicGraphConvolution(nn.Module):
    """Mix predefined and state-conditioned propagation up to ``depth`` hops."""

    def __init__(self, input_dim: int, output_dim: int, depth: int, static_supports: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("static_supports", static_supports)
        terms = 1 + 4 * depth
        self.projection = nn.Linear(input_dim * terms, output_dim)

    def forward(self, x: torch.Tensor, dynamic: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        terms = [x]
        for support in self.static_supports:
            propagated = x
            for _ in range(self.depth):
                propagated = torch.einsum("ij,bjf->bif", support, propagated)
                terms.append(propagated)
        for support in dynamic:
            propagated = x
            for _ in range(self.depth):
                propagated = torch.einsum("bij,bjf->bif", support, propagated)
                terms.append(propagated)
        return self.projection(self.dropout(torch.cat(terms, dim=-1)))


class DynamicGraphGRUCell(GraphConvGRUCell):
    """Graph GRU over static plus dynamic filters; ``forward(x, hidden, graphs)``."""

    def __init__(self, input_dim: int, hidden: int, depth: int, static: torch.Tensor, dropout: float) -> None:
        gates = DynamicGraphConvolution(input_dim + hidden, 2 * hidden, depth, static, dropout)
        candidate = DynamicGraphConvolution(input_dim + hidden, hidden, depth, static, dropout)
        super().__init__(gates, candidate)
        self.hidden = hidden


class Model(nn.Module):
    """DGCRN encoder and autoregressive decoder under TSFLab contracts."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        gcn_depth: int = 1,
        rnn_size: int = 16,
        node_dim: int = 8,
        hyper_gnn_dim: int = 8,
        middle_dim: int = 2,
        tanhalpha: float = 3.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("lengths, nodes, graph depth, and hidden dimensions must be positive")
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(f"adj_mx must have shape {(num_nodes, num_nodes)}")
        static = torch.stack((_transition(adjacency), _transition(adjacency.T)))
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.graph_generator = DynamicGraphGenerator(
            num_nodes, rnn_size, node_dim, hyper_gnn_dim, middle_dim, tanhalpha
        )
        self.cell = DynamicGraphGRUCell(2, rnn_size, gcn_depth, static, dropout)
        self.projection = nn.Linear(rnn_size, 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        historic_time = _time_driver(x_enc, x_mark_enc, self.seq_len)
        hidden = x_enc.new_zeros((x_enc.shape[0], self.num_nodes, self.cell.hidden))
        for step in range(self.seq_len):
            graphs = self.graph_generator(hidden)
            recurrent_input = torch.cat((x_enc[:, step, :, None], historic_time[:, step]), -1)
            hidden = self.cell(recurrent_input, hidden, graphs)

        future_time = _time_driver(x_enc, x_mark_dec, self.pred_len)
        previous = x_enc[:, -1, :, None]
        outputs = []
        for step in range(self.pred_len):
            graphs = self.graph_generator(hidden)
            hidden = self.cell(torch.cat((previous, future_time[:, step]), -1), hidden, graphs)
            previous = self.projection(hidden)
            outputs.append(previous[..., 0])
        return torch.stack(outputs, dim=1)
