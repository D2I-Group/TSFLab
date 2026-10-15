"""Clean-room BiST implementation derived from the PVLDB paper equations."""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.marks import normalized_time_features
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._inputs import check_window


class ResidualMLP(nn.Module):
    """Equation 9/16 residual feed-forward layer."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class Model(nn.Module):
    """Forward base prediction plus backward residual correction (Eq. 5--24)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        model_dim: int = 32,
        prompt_dim: int = 16,
        num_layers: int = 3,
        tod_size: int = 24,
        kernel_size: int = 3,
        residual_steps: int = 2,
        graph_dim: int = 8,
        virtual_clusters: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.tod_size = tod_size
        self.residual_steps = residual_steps

        # Equations 5--7: edge-padded stable/trend split and separate MLP maps.
        self.decomposition = SeriesDecomposition(kernel_size)
        self.stable_projection = nn.Linear(seq_len, model_dim)
        self.trend_projection = nn.Linear(seq_len, model_dim)

        # Equation 8: temporal and spatial prompts.
        self.node_prompt = nn.Parameter(torch.empty(enc_in, prompt_dim))
        self.time_prompt = nn.Embedding(tod_size, prompt_dim)
        self.weekday_prompt = nn.Embedding(7, prompt_dim)
        width = model_dim + 3 * prompt_dim
        self.forward_layers = nn.ModuleList([ResidualMLP(width) for _ in range(num_layers)])
        self.base_head = nn.Linear(width, pred_len)

        # Equations 11--16: virtual-cluster context and personalized residual.
        self.node_queries = nn.Parameter(torch.empty(enc_in, width))
        self.cluster_keys = nn.Parameter(torch.empty(virtual_clusters, width))
        self.residual_alignment = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.residual_layers = nn.ModuleList([ResidualMLP(width) for _ in range(num_layers)])

        # Equations 17--22: adaptive diffusion with node-wise alpha and beta.
        self.graph_embedding = nn.Parameter(torch.empty(enc_in, graph_dim))
        self.alpha = nn.Parameter(torch.zeros(enc_in))
        self.beta = nn.Parameter(torch.zeros(enc_in))
        self.correction_head = nn.Linear(width, pred_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.node_prompt,
            self.node_queries,
            self.cluster_keys,
            self.graph_embedding,
        ):
            nn.init.xavier_uniform_(parameter)
        nn.init.constant_(self.alpha, 0.1)

    def _calendar_indices(self, x: torch.Tensor, marks: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        if marks is None:
            return (
                torch.zeros(batch, dtype=torch.long, device=x.device),
                torch.zeros(batch, dtype=torch.long, device=x.device),
            )
        if marks.ndim == 4:
            if marks.shape[:3] != x.shape:
                raise ValueError("node covariates must have shape [batch, time, nodes, features]")
            if marks.shape[-1] < 2:
                raise ValueError("BiST requires time-of-day and weekday covariates")
            calendar = marks[:, -1, :, :2].mean(dim=1)
        elif marks.ndim == 3 and marks.shape[:2] == x.shape[:2] and marks.shape[-1] == 6:
            calendar = normalized_time_features(marks)[:, -1]
        else:
            raise ValueError("marks must be raw [B,T,6] or node covariates [B,T,N,F]")
        time_index = torch.floor(calendar[:, 0] * self.tod_size).long().clamp(0, self.tod_size - 1)
        weekday_index = torch.floor(calendar[:, 1] * 7).long().clamp(0, 6)
        return time_index, weekday_index

    def _adaptive_kernel(self) -> torch.Tensor:
        affinity = torch.relu(self.graph_embedding @ self.graph_embedding.T)
        affinity = affinity.masked_fill(
            torch.eye(self.enc_in, dtype=torch.bool, device=affinity.device),
            float("-inf"),
        )
        adjacency = torch.zeros_like(affinity) if self.enc_in == 1 else affinity.softmax(-1)
        return torch.diag(torch.sigmoid(self.beta)) + (torch.diag(torch.tanh(self.alpha)) @ adjacency)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        stable, trend = self.decomposition(x_enc)
        temporal = self.stable_projection(stable.transpose(1, 2))
        temporal = temporal + self.trend_projection(trend.transpose(1, 2))
        time_index, weekday_index = self._calendar_indices(x_enc, x_mark_enc)
        batch = x_enc.shape[0]
        prompts = torch.cat(
            (
                temporal,
                self.node_prompt.unsqueeze(0).expand(batch, -1, -1),
                self.time_prompt(time_index).unsqueeze(1).expand(-1, self.enc_in, -1),
                self.weekday_prompt(weekday_index).unsqueeze(1).expand(-1, self.enc_in, -1),
            ),
            dim=-1,
        )
        label_representation = prompts
        for layer in self.forward_layers:
            label_representation = layer(label_representation)
        base = self.base_head(label_representation).transpose(1, 2)

        memberships = (self.node_queries @ self.cluster_keys.T / math.sqrt(self.node_queries.shape[-1])).softmax(-1)
        context_graph = memberships @ memberships.T
        context_graph = context_graph / context_graph.sum(-1, keepdim=True).clamp_min(1e-6)
        common = torch.einsum("nm,bmd->bnd", context_graph, label_representation)
        personalized = label_representation - common
        aligned = self.residual_alignment(torch.cat((personalized, common), dim=-1))
        residual = prompts - aligned
        for layer in self.residual_layers:
            residual = layer(residual)
        kernel = self._adaptive_kernel()
        for _ in range(self.residual_steps):
            residual = torch.einsum("nm,bmd->bnd", kernel, residual)
        correction = self.correction_head(residual).transpose(1, 2)
        return base + correction
