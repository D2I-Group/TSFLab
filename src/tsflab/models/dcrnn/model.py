"""Independent DCRNN implementation from the ICLR 2018 equations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.channel_alignment import fit_channels
from tsflab.models._components.graph_conv_gru import GraphConvGRUCell
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


class DiffusionConvolution(nn.Module):
    """Equation (2): bidirectional random-walk Chebyshev diffusion filters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        diffusion_order: int,
        supports: torch.Tensor,
    ) -> None:
        super().__init__()
        self.diffusion_order = diffusion_order
        self.register_buffer("supports", supports)
        terms = 1 + supports.shape[0] * diffusion_order
        self.projection = nn.Linear(input_dim * terms, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        terms = [x]
        for support in self.supports:
            if self.diffusion_order == 0:
                continue
            previous = x
            current = torch.einsum("ij,bjf->bif", support, x)
            terms.append(current)
            for _ in range(2, self.diffusion_order + 1):
                following = 2.0 * torch.einsum("ij,bjf->bif", support, current) - previous
                terms.append(following)
                previous, current = current, following
        return self.projection(torch.cat(terms, dim=-1))


class DCGRUCell(GraphConvGRUCell):
    """GRU gates whose affine maps are replaced by diffusion convolution.

    The supports are buffers of the two filters, so ``forward(x, hidden)`` takes
    no graph argument.
    """

    def __init__(self, input_dim: int, hidden_dim: int, order: int, supports: torch.Tensor) -> None:
        gates = DiffusionConvolution(input_dim + hidden_dim, 2 * hidden_dim, order, supports)
        candidate = DiffusionConvolution(input_dim + hidden_dim, hidden_dim, order, supports)
        super().__init__(gates, candidate)
        self.hidden_dim = hidden_dim


class RecurrentStack(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, order: int, supports: torch.Tensor) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList(
            DCGRUCell(input_dim if index == 0 else hidden_dim, hidden_dim, order, supports) for index in range(layers)
        )

    def step(self, x: torch.Tensor, state: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        next_state = []
        for cell, hidden in zip(self.cells, state):
            x = cell(x, hidden)
            next_state.append(x)
        return x, next_state

    def zeros(self, batch: int, nodes: int, reference: torch.Tensor) -> list[torch.Tensor]:
        return [reference.new_zeros((batch, nodes, self.hidden_dim)) for _ in self.cells]


class Model(nn.Module):
    """Diffusion-convolutional encoder-decoder without target leakage."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        rnn_units: int = 16,
        num_rnn_layers: int = 1,
        max_diffusion_step: int = 2,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else square_adjacency(adj_mx, num_nodes)
        supports = torch.stack(adj_to_supports(adjacency))
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.encoder = RecurrentStack(input_dim, rnn_units, num_rnn_layers, max_diffusion_step, supports)
        self.decoder = RecurrentStack(1, rnn_units, num_rnn_layers, max_diffusion_step, supports)
        self.projection = nn.Linear(rnn_units, 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.num_nodes)
        history = fit_channels(to_spatiotemporal(x_enc, x_mark_enc), self.input_dim)
        state = self.encoder.zeros(x_enc.shape[0], self.num_nodes, x_enc)
        for step in range(self.seq_len):
            _, state = self.encoder.step(history[:, step], state)
        decoder_input = x_enc.new_zeros((x_enc.shape[0], self.num_nodes, 1))
        outputs = []
        for _ in range(self.pred_len):
            decoded, state = self.decoder.step(decoder_input, state)
            decoder_input = self.projection(decoded)
            outputs.append(decoder_input[..., 0])
        return torch.stack(outputs, dim=1)
