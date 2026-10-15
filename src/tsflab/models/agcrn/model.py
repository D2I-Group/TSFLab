"""Local AGCRN implementation guided by the paper and official reference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.graph_conv_gru import GraphConvGRUCell
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.node_adaptive_graph_conv import NodeAdaptiveGraphConv
from tsflab.models._inputs import check_window


class AdaptiveGraphGRUCell(GraphConvGRUCell):
    """GRU gates and candidate parameterized by adaptive graph filters.

    ``forward(x, state, nodes)`` is the shared ``graph_conv_gru`` recurrence with
    both filters called as ``conv(joined, nodes)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, order: int, node_dim: int) -> None:
        joint = in_dim + hidden_dim
        gates = NodeAdaptiveGraphConv(joint, 2 * hidden_dim, order, node_dim)
        candidate = NodeAdaptiveGraphConv(joint, hidden_dim, order, node_dim)
        super().__init__(gates, candidate)
        self.hidden_dim = hidden_dim


class Model(nn.Module):
    """Adaptive graph convolutional recurrent multi-step forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        rnn_units: int = 32,
        embed_dim: int = 8,
        num_layers: int = 1,
        cheb_k: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        del adj_mx, output_dim
        if num_nodes < 1:
            raise ValueError("AGCRN dimensions must be positive")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.cells = nn.ModuleList(
            AdaptiveGraphGRUCell(
                input_dim if layer == 0 else rnn_units,
                rnn_units,
                cheb_k,
                embed_dim,
            )
            for layer in range(num_layers)
        )
        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, rnn_units))
        self.readout = nn.Sequential(
            nn.Linear(2 * rnn_units, rnn_units),
            nn.GELU(),
            nn.Linear(rnn_units, 1),
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.xavier_uniform_(self.horizon_embedding)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        check_window(x_enc, self.seq_len, self.num_nodes)
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] < self.input_dim:
            raise ValueError("AGCRN received fewer input features than configured")
        states = [x_enc.new_zeros(x_enc.shape[0], self.num_nodes, cell.hidden_dim) for cell in self.cells]
        for step in history[..., : self.input_dim].unbind(1):
            layer_input = step
            for index, cell in enumerate(self.cells):
                states[index] = cell(layer_input, states[index], self.node_embeddings)
                layer_input = states[index]
        final = states[-1].unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        horizon = self.horizon_embedding.view(1, self.pred_len, 1, -1).expand_as(final)
        return self.readout(torch.cat((final, horizon), dim=-1)).squeeze(-1)


__all__ = ["Model", "AdaptiveGraphGRUCell"]
