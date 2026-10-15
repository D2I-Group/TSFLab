"""Independent BigST implementation from the PVLDB method description."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


class Model(nn.Module):
    """Single-stage BigST with positive random-feature spatial attention."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        hid_dim: int = 16,
        node_dim: int = 8,
        time_dim: int = 8,
        tod_size: int = 24,
        dow_size: int = 7,
        tau: float = 1.0,
        random_feature_dim: int = 16,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("lengths and nodes must be positive")
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.tau, self.use_residual = float(tau), use_residual
        self.value_projection = nn.Linear(seq_len * input_dim, hid_dim)
        self.node_source = nn.Parameter(torch.randn(num_nodes, node_dim) / math.sqrt(node_dim))
        self.node_target = nn.Parameter(torch.randn(num_nodes, node_dim) / math.sqrt(node_dim))
        self.time_of_day = nn.Embedding(tod_size, time_dim)
        self.day_of_week = nn.Embedding(dow_size, time_dim)
        context_dim = hid_dim + node_dim + 2 * time_dim
        self.query, self.key = nn.Linear(context_dim, random_feature_dim), nn.Linear(context_dim, random_feature_dim)
        self.value = nn.Linear(context_dim, hid_dim)
        self.random_projection = nn.Parameter(
            torch.randn(random_feature_dim, random_feature_dim) / math.sqrt(random_feature_dim)
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hid_dim) if use_bn else nn.Identity()
        self.forecast = nn.Linear(hid_dim, pred_len)
        adj = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
        adj = adj + np.eye(num_nodes, dtype=np.float32)
        adj /= np.maximum(adj.sum(-1, keepdims=True), 1e-6)
        self.register_buffer("graph_prior", torch.from_numpy(adj))

    @staticmethod
    def _positive_features(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x) + 1.0

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
        base = torch.tanh(self.value_projection(st[..., : self.input_dim].transpose(1, 2).flatten(2)))
        if x_mark_enc is None:
            tod = torch.zeros(x_enc.shape[0], self.num_nodes, dtype=torch.long, device=x_enc.device)
            dow = tod
        else:
            last = st[:, -1, :, 1:]
            tod = (
                ((last[..., 0] * self.time_of_day.num_embeddings).long() % self.time_of_day.num_embeddings)
                if last.shape[-1]
                else torch.zeros_like(x_enc[:, 0], dtype=torch.long)
            )
            dow = (
                (
                    (last[..., min(1, last.shape[-1] - 1)] * self.day_of_week.num_embeddings).long()
                    % self.day_of_week.num_embeddings
                )
                if last.shape[-1]
                else torch.zeros_like(tod)
            )
        source_node = self.node_source.unsqueeze(0).expand(x_enc.shape[0], -1, -1)
        target_node = self.node_target.unsqueeze(0).expand_as(source_node)
        time_context = (self.time_of_day(tod), self.day_of_week(dow))
        source = torch.cat((base, source_node, *time_context), -1)
        target = torch.cat((base, target_node, *time_context), -1)
        projection = self.random_projection / max(self.tau, 1e-6)
        q = self._positive_features(self.query(source) @ projection)
        k = self._positive_features(self.key(target) @ projection)
        v = self.value(source)
        kv = torch.einsum("bnr,bnh->brh", k, v)
        message = torch.einsum("bnr,brh->bnh", q, kv) / torch.einsum("bnr,br->bn", q, k.sum(1)).clamp_min(
            1e-6
        ).unsqueeze(-1)
        message = message + self.prior_scale * torch.einsum("nm,bmh->bnh", self.graph_prior, v)
        if self.use_residual:
            message = message + base
        hidden = self.norm(self.dropout(message).transpose(1, 2)).transpose(1, 2)
        return self.forecast(hidden).transpose(1, 2)
