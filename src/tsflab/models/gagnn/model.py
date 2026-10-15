"""Clean-room GAGNN: city/group graph hierarchy for air forecasting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


def _normalized_graph(adj: np.ndarray, nodes: int) -> torch.Tensor:
    value = square_adjacency(adj, nodes, name="adjacency")
    value = value + np.eye(nodes, dtype=np.float32)
    return torch.from_numpy(value / np.maximum(value.sum(-1, keepdims=True), 1e-6))


class GroupAwareLayer(nn.Module):
    """Message passing between cities and learned latent city groups."""

    def __init__(self, width: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.city_message = nn.Linear(width, width)
        self.group_query = nn.Parameter(torch.randn(groups, width) / width**0.5)
        self.group_message = nn.Linear(width, width)
        self.fusion = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)
        self.last_assignment: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        city = torch.einsum("nm,bmd->bnd", graph, self.city_message(x))
        assignment = torch.softmax(torch.einsum("bnd,gd->bng", x, self.group_query), -1)
        self.last_assignment = assignment
        group = torch.einsum("bng,bnd->bgd", assignment, x) / assignment.sum(1).clamp_min(1e-6).unsqueeze(-1)
        group_affinity = torch.softmax(torch.einsum("bgd,bhd->bgh", group, group) / x.shape[-1] ** 0.5, -1)
        group = torch.einsum("bgh,bhd->bgd", group_affinity, self.group_message(group))
        returned = torch.einsum("bng,bgd->bnd", assignment, group)
        return self.norm(x + self.fusion(torch.cat((x, city, returned), -1)))


class Model(nn.Module):
    """Temporal encoder followed by hierarchical group-aware graph layers."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        group_num: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in, self.cov_dim = seq_len, pred_len, enc_in, cov_dim
        adj = np.eye(enc_in, dtype=np.float32) if adj_mx is None else adj_mx
        self.register_buffer("city_graph", _normalized_graph(adj, enc_in))
        self.input_projection = nn.Linear(1 + cov_dim, d_model)
        self.temporal = nn.GRU(d_model, d_model, batch_first=True)
        self.layers = nn.ModuleList(GroupAwareLayer(d_model, group_num, dropout) for _ in range(num_layers))
        self.head = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        st = to_spatiotemporal(x_enc, x_mark_enc)
        needed = 1 + self.cov_dim
        if st.shape[-1] < needed:
            st = torch.cat((st, st.new_zeros(*st.shape[:-1], needed - st.shape[-1])), -1)
        encoded = self.input_projection(st[..., :needed]).transpose(1, 2)
        batch, nodes, steps, width = encoded.shape
        _, hidden = self.temporal(encoded.reshape(batch * nodes, steps, width))
        state = hidden[-1].reshape(batch, nodes, width)
        for layer in self.layers:
            state = layer(state, self.city_graph)
        return self.head(state).transpose(1, 2)
