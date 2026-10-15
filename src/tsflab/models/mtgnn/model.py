"""Independent MTGNN implementation: graph learning and mix-hop propagation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.gated_dilated_conv import gated_dilated_conv
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


class GraphConstructor(nn.Module):
    def __init__(self, nodes: int, dimension: int, top_k: int, alpha: float) -> None:
        super().__init__()
        self.left, self.right = (
            nn.Parameter(torch.randn(nodes, dimension)),
            nn.Parameter(torch.randn(nodes, dimension)),
        )
        self.left_map, self.right_map = (
            nn.Linear(dimension, dimension),
            nn.Linear(dimension, dimension),
        )
        self.top_k, self.alpha = min(top_k, nodes), alpha

    def forward(self) -> torch.Tensor:
        left, right = (
            torch.tanh(self.alpha * self.left_map(self.left)),
            torch.tanh(self.alpha * self.right_map(self.right)),
        )
        scores = torch.relu(torch.tanh(self.alpha * (left @ right.T - right @ left.T)))
        if self.top_k < scores.shape[0]:
            scores = scores * (scores >= scores.topk(self.top_k, -1).values[..., -1:])
        return scores / scores.sum(-1, keepdim=True).clamp_min(1e-6)


class MixHop(nn.Module):
    def __init__(self, width: int, depth: int, alpha: float) -> None:
        super().__init__()
        self.depth, self.alpha = depth, alpha
        self.project = nn.Linear((depth + 1) * width, width)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        states, current = [x], x
        for _ in range(self.depth):
            current = self.alpha * x + (1 - self.alpha) * torch.einsum("nm,btmd->btnd", graph, current)
            states.append(current)
        return self.project(torch.cat(states, -1))


class MTGNNLayer(nn.Module):
    def __init__(self, width: int, graph_width: int, depth: int, alpha: float, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.filter, self.gate = (
            nn.Conv2d(width, graph_width, (1, 3), dilation=(1, dilation)),
            nn.Conv2d(width, graph_width, (1, 3), dilation=(1, dilation)),
        )
        self.forward_graph, self.backward_graph = (
            MixHop(graph_width, depth, alpha),
            MixHop(graph_width, depth, alpha),
        )
        self.residual, self.norm, self.dropout = (
            nn.Linear(graph_width, width),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        channels = x.permute(0, 3, 2, 1)
        temporal = gated_dilated_conv(channels, self.filter, self.gate).permute(0, 3, 2, 1)
        spatial = self.forward_graph(temporal, graph) + self.backward_graph(temporal, graph.T)
        return self.norm(x + self.dropout(self.residual(spatial)))


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        gcn_depth: int = 2,
        subgraph_size: int = 20,
        node_dim: int = 40,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        layers: int = 3,
        dropout: float = 0.3,
        propalpha: float = 0.05,
        tanhalpha: float = 3.0,
        dilation_exponential: int = 1,
        build_adj: bool = True,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("invalid MTGNN dimensions")
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim, self.build_adj = (
            seq_len,
            pred_len,
            num_nodes,
            input_dim,
            build_adj,
        )
        adj = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
        adj = adj + np.eye(num_nodes, dtype=np.float32)
        adj /= np.maximum(adj.sum(-1, keepdims=True), 1e-6)
        self.register_buffer("predefined_graph", torch.from_numpy(adj))
        self.graph_constructor, self.graph_mix = (
            GraphConstructor(num_nodes, node_dim, subgraph_size, tanhalpha),
            nn.Parameter(torch.tensor(0.5)),
        )
        self.input_projection = nn.Linear(input_dim, residual_channels)
        self.layers = nn.ModuleList(
            MTGNNLayer(
                residual_channels,
                conv_channels,
                gcn_depth,
                propalpha,
                max(1, dilation_exponential**i),
                dropout,
            )
            for i in range(layers)
        )
        self.skip = nn.Linear(residual_channels, skip_channels)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(skip_channels, end_channels),
            nn.GELU(),
            nn.Linear(end_channels, pred_len),
        )

    def learned_graph(self) -> torch.Tensor:
        if not self.build_adj:
            return self.predefined_graph
        weight = torch.sigmoid(self.graph_mix)
        graph = weight * self.graph_constructor() + (1 - weight) * self.predefined_graph
        return graph / graph.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.num_nodes)
        st = to_spatiotemporal(x_enc, x_mark_enc)
        if st.shape[-1] < self.input_dim:
            st = torch.cat((st, st.new_zeros(*st.shape[:-1], self.input_dim - st.shape[-1])), -1)
        state = self.input_projection(st[..., : self.input_dim])
        graph = self.learned_graph()
        for layer in self.layers:
            state = layer(state, graph)
        return self.head(self.skip(state[:, -1])).transpose(1, 2)
