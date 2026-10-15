"""Local HimNet implementation from paper and official-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.graph_conv_gru import GraphConvGRUCell
from tsflab.models._components.marks import normalized_time_features, to_spatiotemporal
from tsflab.models._components.node_adaptive_graph_conv import NodeAdaptiveGraphConv
from tsflab.models._inputs import check_window


class MetaGraphGRUCell(GraphConvGRUCell):
    """Graph GRU over meta-parameterized filters; ``forward(x, state, meta)``."""

    def __init__(self, in_dim: int, hidden: int, order: int, meta_dim: int) -> None:
        gates = NodeAdaptiveGraphConv(in_dim + hidden, 2 * hidden, order, meta_dim)
        candidate = NodeAdaptiveGraphConv(in_dim + hidden, hidden, order, meta_dim)
        super().__init__(gates, candidate)
        self.hidden = hidden


class Model(nn.Module):
    """Hierarchical meta-parameterized encoder-decoder graph network."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        cheb_k: int = 2,
        node_embedding_dim: int = 8,
        st_embedding_dim: int = 8,
        tod_embedding_dim: int = 8,
        dow_embedding_dim: int = 8,
        steps_per_day: int = 288,
        use_teacher_forcing: bool = True,
    ) -> None:
        super().__init__()
        del adj_mx, use_teacher_forcing
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.steps_per_day = steps_per_day
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, st_embedding_dim))
        context_dim = node_embedding_dim + tod_embedding_dim + dow_embedding_dim + st_embedding_dim
        self.meta_projection = nn.Linear(context_dim, node_embedding_dim)
        self.encoder = nn.ModuleList(
            MetaGraphGRUCell(input_dim if layer == 0 else hidden_dim, hidden_dim, cheb_k, node_embedding_dim)
            for layer in range(num_layers)
        )
        self.decoder = nn.ModuleList(
            MetaGraphGRUCell(1 if layer == 0 else hidden_dim, hidden_dim, cheb_k, node_embedding_dim)
            for layer in range(num_layers)
        )
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.horizon_embedding)

    def _meta(self, tod: torch.Tensor, dow: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        batch, nodes = tod.shape
        pieces = [
            self.node_embedding.unsqueeze(0).expand(batch, -1, -1),
            self.tod_embedding(tod),
            self.dow_embedding(dow),
            horizon.view(1, 1, -1).expand(batch, nodes, -1),
        ]
        return self.meta_projection(torch.cat(pieces, -1))

    @staticmethod
    def _step(
        cells: nn.ModuleList, value: torch.Tensor, states: list[torch.Tensor], meta: torch.Tensor
    ) -> torch.Tensor:
        for layer, cell in enumerate(cells):
            states[layer] = cell(value, states[layer], meta)
            value = states[layer]
        return value

    def _future_features(self, x_mark_dec: torch.Tensor | None) -> torch.Tensor | None:
        return (
            normalized_time_features(x_mark_dec[:, -self.pred_len :])
            if x_mark_dec is not None and x_mark_dec.ndim == 3
            else None
        )

    def _future_calendar(
        self, future: torch.Tensor | None, horizon: int, x_enc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if future is None:
            tod = torch.full(
                (x_enc.shape[0], self.num_nodes), horizon % self.steps_per_day, device=x_enc.device, dtype=torch.long
            )
            return tod, torch.zeros_like(tod)
        tod = (
            (future[:, horizon, 0:1] * self.steps_per_day)
            .long()
            .expand(-1, self.num_nodes)
            .clamp(0, self.steps_per_day - 1)
        )
        dow = (future[:, horizon, 1:2] * 7).long().expand(-1, self.num_nodes).clamp(0, 6)
        return tod, dow

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        check_window(x_enc, self.seq_len, self.num_nodes)
        data = to_spatiotemporal(x_enc, x_mark_enc)
        states = [x_enc.new_zeros(x_enc.shape[0], self.num_nodes, cell.hidden) for cell in self.encoder]
        zero_horizon = self.horizon_embedding.new_zeros(self.horizon_embedding.shape[-1])
        for index, step in enumerate(data[..., : self.input_dim].unbind(1)):
            tod = (data[:, index, :, 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            dow = (data[:, index, :, 2] * 7).long().clamp(0, 6)
            self._step(self.encoder, step, states, self._meta(tod, dow, zero_horizon))
        decoder_input = x_enc[:, -1].unsqueeze(-1)
        outputs = []
        future = self._future_features(x_mark_dec)
        for horizon in range(self.pred_len):
            tod, dow = self._future_calendar(future, horizon, x_enc)
            meta = self._meta(tod, dow, self.horizon_embedding[horizon])
            decoder_input = self.output(self._step(self.decoder, decoder_input, states, meta))
            outputs.append(decoder_input.squeeze(-1))
        return torch.stack(outputs, dim=1)


__all__ = ["Model", "MetaGraphGRUCell"]
