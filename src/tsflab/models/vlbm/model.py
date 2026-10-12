"""Local VLBM: variational latent-basis activations with a base/residual generator.

Independent implementation from the paper (Section IV, Eqs. 17-25 and 28-33) and
the pinned official code (``leijieruilq/VLBM_OOD_forecast``). The forecast path
follows the official ``models/VLBM.py``: the prior-sampled latent state ``Z`` drives
a base stream built from ``[Z, X]`` and a residual stream built from ``[X - Z, X]``
propagated over a top-k cosine graph of ``Z``; the paper's ridge projection onto the
basis span (Eqs. 26-27) is not part of the official code and is not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import to_calendar_spatiotemporal
from tsflab.models._components.revin import RevIN

#: Fixed constants of the official code.
BLOCK_DROPOUT = 0.15
GRAPH_TOPOLOGY_WEIGHT = 0.2
LEAKY_SLOPE = 0.2
MASK_FILL = -1e9


def calendar_indices(values: torch.Tensor, marks, steps_per_day: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Time-of-day and day-of-week slots ``[B, N]`` of the last step of a window.

    ``marks`` are raw ``[B, T, 6]`` calendar marks, ``[B, T, N, 2]`` normalized
    calendar covariates, or ``None`` (slot 0). Slots are ``floor(frac * size)``
    with a small guard against float rounding just below an integer.
    """
    features = to_calendar_spatiotemporal(values, marks)[:, -1]
    tod = torch.floor(features[..., 1] * steps_per_day + 1e-3).long().clamp(0, steps_per_day - 1)
    dow = torch.floor(features[..., 2] * 7 + 1e-3).long().clamp(0, 6)
    return tod, dow


class VariableEmbedding(nn.Module):
    """Eq. 17: ``X = Conv_emb([E_time, E_day, E_val, E_node])`` as ``[B, N, d]``."""

    def __init__(self, num_nodes: int, hidden: int, length: int, steps_per_day: int) -> None:
        super().__init__()
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, hidden))
        self.time_embedding = nn.Parameter(torch.empty(steps_per_day, hidden))
        self.day_embedding = nn.Parameter(torch.empty(7, hidden))
        for table in (self.node_embedding, self.time_embedding, self.day_embedding):
            nn.init.xavier_uniform_(table)
        # 1x1 convolutions of the official code are pointwise linear maps.
        self.value_projection = nn.Linear(length, hidden)
        self.fuse = nn.Linear(4 * hidden, hidden)

    def forward(self, values: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        pieces = [
            self.time_embedding[tod],
            self.day_embedding[dow],
            self.value_projection(values.transpose(1, 2)),
            self.node_embedding.unsqueeze(0).expand(batch, -1, -1),
        ]
        return self.fuse(torch.cat(pieces, dim=-1))


class GraphStructuredEncoder(nn.Module):
    """Eqs. 20-23: masked additive graph attention followed by Gaussian heads.

    ``u`` is the query side and ``v`` the value side (``(X, Y)`` for the
    posterior, ``(X, X)`` for the prior); edges outside ``adjacency > 0`` are
    filled with ``-1e9`` before the softmax over ``j``.
    """

    def __init__(self, hidden: int, num_basis: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.score = nn.Linear(2 * hidden, 1)
        self.mu = nn.Linear(hidden, num_basis)
        self.logvar = nn.Linear(hidden, num_basis)

    def aggregate(self, u: torch.Tensor, v: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = u.shape[-1]
        query = F.relu(self.query(u))
        value = F.relu(self.value(v))
        # e_ij = LeakyReLU(a^T [u_i, v_j] + b), split into its query and value halves.
        left = query @ self.score.weight[:, :hidden].T
        right = value @ self.score.weight[:, hidden:].T
        scores = F.leaky_relu(left + right.transpose(1, 2) + self.score.bias, LEAKY_SLOPE)
        scores = torch.where(adjacency > 0, scores, torch.full_like(scores, MASK_FILL))
        return scores.softmax(dim=-1) @ value

    def forward(self, u, v, adjacency) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.aggregate(u, v, adjacency)
        return self.mu(h), self.logvar(h)


def sample_latent(mu: torch.Tensor, logvar: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Eqs. 19 and 24-25: ``w = mu + exp(logvar / 2) * eps``, ``z = w B``."""
    weights = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    return weights @ basis


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """``KL(N(mu_q, var_q) || N(mu_p, var_p))`` summed over the basis axis, averaged over batch and variables."""
    elementwise = 0.5 * (
        logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p).square()) / torch.exp(logvar_p) - 1.0
    )
    return elementwise.sum(dim=-1).mean()


def latent_graph(z: torch.Tensor, top_k: int, alpha: float = GRAPH_TOPOLOGY_WEIGHT) -> torch.Tensor:
    """Eq. 31 with the identity topology: ``Softmax(TopK((1 - a) cos(z_i, z_j) + a I))``."""
    unit = F.normalize(z, dim=-1)
    similarity = unit @ unit.transpose(1, 2)
    eye = torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    similarity = (1.0 - alpha) * similarity + alpha * eye
    keep = torch.zeros_like(similarity).scatter(-1, similarity.topk(top_k, dim=-1).indices, 1.0)
    return similarity.masked_fill(keep == 0, MASK_FILL).softmax(dim=-1)


class PointwiseResidualBlock(nn.Module):
    """Eq. 29: ``P + W2 Drop(ReLU(W1 P))`` over the hidden axis (optionally on ``A P``)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.inner = nn.Linear(hidden, hidden)
        self.outer = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(BLOCK_DROPOUT)

    def forward(self, state: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        mixed = state if adjacency is None else adjacency @ state
        return state + self.outer(self.dropout(F.relu(self.inner(mixed))))


class BaseResidualGenerator(nn.Module):
    """Base stream from ``[Z, X]`` and graph-propagated residual stream from ``[X - Z, X]``."""

    def __init__(self, hidden: int, num_blocks: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.base_input = nn.Linear(2 * hidden, hidden)
        self.residual_input = nn.Linear(2 * hidden, hidden)
        self.base_blocks = nn.ModuleList(PointwiseResidualBlock(hidden) for _ in range(num_blocks))
        self.residual_blocks = nn.ModuleList(PointwiseResidualBlock(hidden) for _ in range(num_blocks))

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_input(torch.cat([x - z, x], dim=-1))
        base = self.base_input(torch.cat([z, x], dim=-1))
        for block in self.base_blocks:
            base = block(base)
        graph = latent_graph(z, self.top_k)
        for block in self.residual_blocks:
            residual = block(residual, graph)
        return base + residual


@dataclass
class VariationalOutput:
    forecast: torch.Tensor
    mu_p: torch.Tensor
    logvar_p: torch.Tensor
    mu_q: torch.Tensor | None = None
    logvar_q: torch.Tensor | None = None

    def kl(self) -> torch.Tensor:
        if self.mu_q is None:
            raise ValueError("the posterior was not evaluated")
        return gaussian_kl(self.mu_q, self.logvar_q, self.mu_p, self.logvar_p)


class Model(nn.Module):
    """VLBM forecaster for ``[B, seq_len, enc_in]`` histories with calendar marks."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden: int = 256,
        num_blocks: int = 2,
        top_k: int = 8,
        num_basis: int = 800,
        interval: float = 5.0,
        use_norm: bool = True,
        kl_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("VLBM sizes must be positive")
        steps_per_day = int(1440 / interval)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.steps_per_day = steps_per_day
        self.kl_weight = float(kl_weight)
        self.norm = RevIN(enc_in, eps=1e-5, affine=False, enabled=use_norm)
        self.history_embedding = VariableEmbedding(enc_in, hidden, seq_len, steps_per_day)
        self.future_embedding = VariableEmbedding(enc_in, hidden, pred_len, steps_per_day)
        self.prior = GraphStructuredEncoder(hidden, num_basis)
        self.posterior = GraphStructuredEncoder(hidden, num_basis)
        self.basis = nn.Parameter(torch.empty(num_basis, hidden))
        nn.init.xavier_uniform_(self.basis)
        self.generator = BaseResidualGenerator(hidden, num_blocks, top_k)
        self.head = nn.Linear(hidden, pred_len)

    def _identity(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.enc_in, dtype=reference.dtype, device=reference.device)

    def _check(self, values: torch.Tensor, length: int, what: str) -> None:
        if values.ndim != 3 or values.shape[1:] != (length, self.enc_in):
            raise ValueError(f"VLBM expects {what} shaped [B, {length}, {self.enc_in}]")

    def variational_forward(self, x_enc, x_mark_enc=None, y_future=None, y_mark_future=None) -> VariationalOutput:
        """Forecast from the prior; also evaluate the posterior when ``y_future`` is given."""
        self._check(x_enc, self.seq_len, "histories")
        identity = self._identity(x_enc)
        tod, dow = calendar_indices(x_enc, x_mark_enc, self.steps_per_day)
        x = self.history_embedding(self.norm(x_enc, "norm"), tod, dow)
        mu_p, logvar_p = self.prior(x, x, identity)
        out = VariationalOutput(forecast=x_enc.new_empty(0), mu_p=mu_p, logvar_p=logvar_p)
        if y_future is not None:
            self._check(y_future, self.pred_len, "futures")
            # The official posterior embeds the future values without the history normalization.
            tod_y, dow_y = calendar_indices(y_future, y_mark_future, self.steps_per_day)
            y = self.future_embedding(y_future, tod_y, dow_y)
            out.mu_q, out.logvar_q = self.posterior(x, y, identity)
        z = sample_latent(mu_p, logvar_p, self.basis)
        stream = self.generator(z, x)
        forecast = self.head(stream).transpose(1, 2)
        out.forecast = self.norm(forecast, "denorm")
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        return self.variational_forward(x_enc, x_mark_enc).forecast


__all__ = [
    "BaseResidualGenerator",
    "GraphStructuredEncoder",
    "Model",
    "PointwiseResidualBlock",
    "VariableEmbedding",
    "VariationalOutput",
    "calendar_indices",
    "gaussian_kl",
    "latent_graph",
    "sample_latent",
]
