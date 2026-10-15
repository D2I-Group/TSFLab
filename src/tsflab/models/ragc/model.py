"""Local RAGC implementation from the paper and pinned reference details.

RAGC ("Efficient Traffic Forecasting on Large-Scale Road Network by
Regularized Adaptive Graph Convolution", arXiv:2506.07179) embeds each
lookback window with a per-node linear layer, concatenates it with a
learnable (optionally row-swap-regularized) node embedding and time-of-day /
day-of-week embeddings, and runs the result through a stack of feed-forward +
adaptive-graph-convolution blocks. Each block's graph convolution is the
paper's "Efficient Cosine Operator" (ECO): a gated, cosine-normalized support
built from the node embeddings, diffused with an associative kernel trick
that never materializes a dense ``(N, N)`` adjacency (linear time in the
number of nodes ``N``). Following the official residual-difference wiring,
each block subtracts the graph-smoothed signal from the running hidden state
(keeping the "high-frequency" residual for the next block) while accumulating
the smoothed signal in a separate skip path; both the final residual state
and the accumulated skip are projected to the forecast horizon and summed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.regularized_adaptive_graph_conv import (
    EfficientCosineGraphConv,
    StochasticSharedEmbedding,
)
from tsflab.models._inputs import check_window, square_adjacency


class FeedForward(nn.Module):
    """Per-node two-layer MLP with a residual connection (paper's block front-end)."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = dim * 2
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(x))))


class Model(nn.Module):
    """RAGC: stochastic-regularized, node-embedding adaptive graph convolution."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        spatial_dim: int = 16,
        embed_dim: int = 16,
        temp_dim_tid: int = 8,
        temp_dim_diw: int = 8,
        num_time_in_day: int = 24,
        num_day_in_week: int = 7,
        num_layer: int = 2,
        order: int = 1,
        sse_p: float = 0.1,
        use_sse: bool = True,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("RAGC num_nodes must be positive")

        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
        self.register_buffer("static_adj", torch.as_tensor(adjacency, dtype=torch.float32))

        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim = input_dim
        self.num_time_in_day = num_time_in_day
        self.num_day_in_week = num_day_in_week
        self.num_layer = num_layer

        self.time_series_emb_layer = nn.Linear(seq_len * input_dim, embed_dim)

        self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.sse = StochasticSharedEmbedding(p=sse_p) if use_sse else nn.Identity()

        self.time_in_day_emb = nn.Parameter(torch.empty(num_time_in_day, temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(num_day_in_week, temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.hidden_dim = spatial_dim + temp_dim_tid + temp_dim_diw + embed_dim
        self.feed_forwards = nn.ModuleList(FeedForward(self.hidden_dim) for _ in range(num_layer))
        self.gconvs = nn.ModuleList(
            EfficientCosineGraphConv(hidden_dim=self.hidden_dim, spatial_dim=spatial_dim, order=order)
            for _ in range(num_layer)
        )

        self.regression_layer = nn.Linear(self.hidden_dim, pred_len)
        self.regression_skip = nn.Linear(self.hidden_dim, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.num_nodes)
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] < self.input_dim:
            raise ValueError("RAGC received fewer input features than configured")
        batch_size = x_enc.shape[0]

        window = history[..., : self.input_dim].transpose(1, 2).reshape(batch_size, self.num_nodes, -1)
        time_series_emb = self.time_series_emb_layer(window)

        node_emb = self.sse(self.node_emb)  # (N, spatial_dim), regularized only in training
        node_emb_batch = node_emb.unsqueeze(0).expand(batch_size, -1, -1)

        latest = history[:, -1]  # (B, N, F): last observed step's calendar covariates
        time_index = (latest[..., 1] * self.num_time_in_day).long().clamp(0, self.num_time_in_day - 1)
        day_index = (latest[..., 2] * self.num_day_in_week).long().clamp(0, self.num_day_in_week - 1)
        time_in_day_emb = self.time_in_day_emb[time_index]
        day_in_week_emb = self.day_in_week_emb[day_index]

        hidden = torch.cat([time_series_emb, node_emb_batch, time_in_day_emb, day_in_week_emb], dim=-1)

        skips = 0.0
        for feed_forward, gconv in zip(self.feed_forwards, self.gconvs):
            hidden = feed_forward(hidden)
            smoothed = gconv(hidden, node_emb)
            hidden = hidden - smoothed  # residual-difference mechanism: keep the high-frequency part
            skips = skips + smoothed

        regression = self.regression_layer(hidden) + self.regression_skip(skips)  # (B, N, pred_len)
        return regression.transpose(1, 2).contiguous()  # (B, pred_len, N)

    def graph_regularization_loss(self) -> torch.Tensor:
        """Optional Laplacian-smoothness regularizer on the node embedding (official ``lap_sm`` ablation).

        **Not** part of the ``forward`` runner contract. The official
        ``main.py`` only adds this term to the training loss
        (``loss = loss_mae + loss_lap * 0.001``) when the ``--lap_sm`` flag
        is passed; none of the paper's released per-dataset commands (SD,
        GBA, GLA, CA) enable it, so it is an optional ablation rather than
        RAGC's core mechanism (that role belongs to
        :class:`~models._components.regularized_adaptive_graph_conv.StochasticSharedEmbedding`,
        wired into ``forward`` above). It is exposed here as a separate
        method — following the same pattern as ST-SSDL's auxiliary losses —
        for a caller that wants to reproduce the ablation from a custom
        training script; this repository's runner does not call it.

        Returns:
            A non-negative scalar: ``trace(E^T L E)`` where ``L`` is the
            symmetric-normalized Laplacian of the (possibly asymmetric)
            static adjacency and ``E`` is the raw node embedding.
        """
        adj = self.static_adj
        adj = (adj.t() + adj) / 2
        row_sum = adj.sum(dim=1)
        laplacian = torch.diag(row_sum) - adj
        inv_sqrt_degree = (row_sum + 1e-3).pow(-0.5)
        inv_sqrt_degree[torch.isinf(inv_sqrt_degree)] = 0.0
        norm = torch.diag(inv_sqrt_degree)
        normalized_laplacian = norm @ laplacian @ norm
        return torch.trace(self.node_emb.t() @ normalized_laplacian @ self.node_emb)


__all__ = ["Model", "FeedForward"]
