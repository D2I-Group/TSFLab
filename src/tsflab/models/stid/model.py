"""Local STID implementation from the paper and pinned reference details."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window


class ResidualPointwiseBlock(nn.Module):
    """Two pointwise transforms used by STID's residual encoder."""

    def __init__(self, width: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.input = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.output(self.dropout(self.activation(self.input(x))))


class Model(nn.Module):
    """Spatial-temporal identity embedding forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        embed_dim: int = 32,
        num_layers: int = 1,
        num_time_in_day: int = 24,
        num_day_in_week: int = 7,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
    ) -> None:
        super().__init__()
        del adj_mx
        if num_nodes < 1:
            raise ValueError("STID dimensions must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_time_in_day = num_time_in_day
        self.num_day_in_week = num_day_in_week
        self.input_projection = nn.Linear(seq_len * input_dim, embed_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.time_embedding = nn.Embedding(num_time_in_day, embed_dim) if if_time_in_day else None
        self.weekday_embedding = nn.Embedding(num_day_in_week, embed_dim) if if_day_in_week else None
        identity_width = embed_dim * (2 + int(if_time_in_day) + int(if_day_in_week))
        self.context_projection = nn.Linear(identity_width, embed_dim)
        self.encoder = nn.Sequential(*[ResidualPointwiseBlock(embed_dim) for _ in range(num_layers)])
        self.forecast = nn.Linear(embed_dim, pred_len)
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.num_nodes)
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] < self.input_dim:
            raise ValueError("STID received fewer input features than configured")
        node_history = history[..., : self.input_dim].transpose(1, 2).flatten(2)
        pieces = [
            self.input_projection(node_history),
            self.node_embedding.unsqueeze(0).expand(x_enc.shape[0], -1, -1),
        ]
        latest = history[:, -1]
        if self.time_embedding is not None:
            index = (latest[..., 1] * self.num_time_in_day).long()
            pieces.append(self.time_embedding(index.clamp(0, self.num_time_in_day - 1)))
        if self.weekday_embedding is not None:
            index = (latest[..., 2] * self.num_day_in_week).long()
            pieces.append(self.weekday_embedding(index.clamp(0, self.num_day_in_week - 1)))
        encoded = self.context_projection(torch.cat(pieces, dim=-1))
        encoded = self.encoder(encoded)
        return self.forecast(encoded).transpose(1, 2).contiguous()


__all__ = ["Model", "ResidualPointwiseBlock"]
