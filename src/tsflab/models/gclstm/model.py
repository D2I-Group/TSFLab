"""Independent graph-convolutional LSTM forecaster."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.channel_alignment import fit_channels
from tsflab.models._components.graph_spectral import chebyshev_supports
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


class ChebyshevGraphProjection(nn.Module):
    """Project concatenated Chebyshev responses of node features."""

    def __init__(self, input_dim: int, output_dim: int, supports: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("supports", supports)
        self.projection = nn.Linear(input_dim * supports.shape[0], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = [torch.einsum("ij,bjf->bif", support, x) for support in self.supports]
        return self.projection(torch.cat(responses, dim=-1))


class GraphConvLSTMCell(nn.Module):
    """LSTM gates produced by Chebyshev graph convolution of input and state."""

    def __init__(self, input_dim: int, hidden_dim: int, supports: torch.Tensor) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = ChebyshevGraphProjection(input_dim + hidden_dim, 4 * hidden_dim, supports)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((x, hidden), dim=-1)
        ).chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class Model(nn.Module):
    """Chebyshev GCLSTM encoder with a direct node-wise horizon readout."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        Ks: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        adjacency = np.ones((enc_in, enc_in), dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, enc_in)
        supports = chebyshev_supports(adjacency, Ks)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = enc_in
        self.input_dim = 1 + cov_dim
        self.cell = GraphConvLSTMCell(self.input_dim, hidden_dim, supports)
        self.forecast = nn.Linear(hidden_dim, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.num_nodes)
        history = fit_channels(to_spatiotemporal(x_enc, x_mark_enc), self.input_dim)
        hidden = history.new_zeros((history.shape[0], self.num_nodes, self.cell.hidden_dim))
        cell = torch.zeros_like(hidden)
        for step in range(self.seq_len):
            hidden, cell = self.cell(history[:, step], hidden, cell)
        return self.forecast(hidden).transpose(1, 2)
