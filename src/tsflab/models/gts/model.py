"""Clean-room GTS: discrete graph discovery plus diffusion recurrence."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.channel_alignment import fit_channels
from tsflab.models._components.graph_conv_gru import GraphConvGRUCell
from tsflab.models._components.marks import to_spatiotemporal


def _row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    adjacency = adjacency + identity
    return adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-8)


class DiscreteGraphDiscovery(nn.Module):
    """Parameterise Bernoulli edges from node histories and sample them."""

    def __init__(self, seq_len: int, nodes: int, embedding_dim: int, temperature: float, prior: torch.Tensor, prior_strength: float) -> None:
        super().__init__()
        self.nodes = nodes
        self.temperature = temperature
        self.prior_strength = prior_strength
        self.history_encoder = nn.Sequential(
            nn.Linear(seq_len, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.node_identity = nn.Parameter(torch.randn(nodes, embedding_dim) / math.sqrt(embedding_dim))
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, 2),
        )
        self.register_buffer("prior", prior)
        self.last_edge_probabilities: torch.Tensor | None = None

    def logits(self, history: torch.Tensor) -> torch.Tensor:
        # history: B,T,N -> B,N,T. A node's observed feature series is Eq. (3)'s input.
        encoded = self.history_encoder(history.transpose(1, 2)) + self.node_identity[None]
        senders = encoded[:, :, None].expand(-1, -1, self.nodes, -1)
        receivers = encoded[:, None, :].expand(-1, self.nodes, -1, -1)
        logits = self.edge_classifier(torch.cat((senders, receivers), dim=-1))
        if self.prior_strength:
            signed_prior = (2.0 * self.prior - 1.0) * self.prior_strength
            logits = logits + torch.stack((-signed_prior, signed_prior), dim=-1)
        return logits

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        logits = self.logits(history)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        self.last_edge_probabilities = probabilities[..., 1]
        if self.training:
            edge_samples = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)[..., 1]
        else:
            edge_samples = probabilities[..., 1]
        diagonal = torch.eye(self.nodes, dtype=torch.bool, device=history.device)
        return edge_samples.masked_fill(diagonal[None], 0.0)

    def prior_loss(self, history: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(self.logits(history), dim=-1)[..., 1].clamp(1e-6, 1 - 1e-6)
        target = self.prior.expand_as(probability)
        mask = ~torch.eye(self.nodes, dtype=torch.bool, device=history.device)
        return F.binary_cross_entropy(probability[:, mask], target[:, mask])


class LearnedDiffusion(nn.Module):
    """DCRNN-style bidirectional polynomial diffusion on a sampled graph."""

    def __init__(self, input_dim: int, output_dim: int, order: int) -> None:
        super().__init__()
        self.order = order
        self.projection = nn.Linear(input_dim * (1 + 2 * order), output_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        supports = (_row_normalize(adjacency), _row_normalize(adjacency.transpose(-1, -2)))
        terms = [x]
        for support in supports:
            previous = x
            current = torch.einsum("bij,bjf->bif", support, x)
            terms.append(current)
            for _ in range(2, self.order + 1):
                following = 2.0 * torch.einsum("bij,bjf->bif", support, current) - previous
                terms.append(following)
                previous, current = current, following
        return self.projection(torch.cat(terms, -1))


class GraphGRUCell(GraphConvGRUCell):
    """Graph GRU over learned-graph diffusion; ``forward(x, hidden, graph)``."""

    def __init__(self, input_dim: int, hidden_dim: int, order: int) -> None:
        gates = LearnedDiffusion(input_dim + hidden_dim, 2 * hidden_dim, order)
        candidate = LearnedDiffusion(input_dim + hidden_dim, hidden_dim, order)
        super().__init__(gates, candidate)
        self.hidden_dim = hidden_dim


class RecurrentStack(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, order: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList(
            GraphGRUCell(input_dim if index == 0 else hidden_dim, hidden_dim, order)
            for index in range(layers)
        )

    def step(self, x: torch.Tensor, state: list[torch.Tensor], graph: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        next_state = []
        for cell, hidden in zip(self.cells, state):
            x = cell(x, hidden, graph)
            next_state.append(x)
        return x, next_state


class Model(nn.Module):
    """Joint discrete graph learner and graph recurrent forecaster."""

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
        embedding_dim: int = 16,
        temp: float = 0.5,
        prior_strength: float = 0.1,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes) < 1:
            raise ValueError("lengths, nodes, channels, and hidden dimensions must be positive")
        if adj_mx is None:
            prior = torch.zeros(num_nodes, num_nodes)
        else:
            adjacency = np.asarray(adj_mx, dtype=np.float32)
            if adjacency.shape != (num_nodes, num_nodes):
                raise ValueError(f"adj_mx must have shape {(num_nodes, num_nodes)}")
            prior = torch.as_tensor((adjacency > 0).astype(np.float32))
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.has_prior = adj_mx is not None
        self.input_dim = input_dim
        self.graph_discovery = DiscreteGraphDiscovery(
            seq_len, num_nodes, embedding_dim, temp, prior, prior_strength
        )
        self.encoder = RecurrentStack(input_dim, rnn_units, num_rnn_layers, max_diffusion_step)
        self.decoder = RecurrentStack(1, rnn_units, num_rnn_layers, max_diffusion_step)
        self.projection = nn.Linear(rnn_units, 1)

    def graph_prior_loss(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.graph_discovery.prior_loss(x_enc)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        graph = self.graph_discovery(x_enc)
        history = fit_channels(to_spatiotemporal(x_enc, x_mark_enc), self.input_dim)
        state = [history.new_zeros((history.shape[0], self.num_nodes, self.encoder.hidden_dim)) for _ in self.encoder.cells]
        for step in range(self.seq_len):
            _, state = self.encoder.step(history[:, step], state, graph)
        previous = x_enc.new_zeros((x_enc.shape[0], self.num_nodes, 1))
        outputs = []
        for _ in range(self.pred_len):
            decoded, state = self.decoder.step(previous, state, graph)
            previous = self.projection(decoded)
            outputs.append(previous[..., 0])
        return torch.stack(outputs, dim=1)
