"""Clean-room PCDCNet following equations (3)--(9) of arXiv:2505.19842."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.marks import TIME_FEATURES, coerce_time_length, future_time_features, to_spatiotemporal
from tsflab.models._inputs import check_window, square_adjacency


def _normalized_laplacian(adjacency: np.ndarray, nodes: int) -> torch.Tensor:
    matrix = square_adjacency(adjacency, nodes)
    degree = matrix.sum(-1)
    scale = np.zeros_like(degree)
    scale[degree > 0] = degree[degree > 0] ** -0.5
    normalized = scale[:, None] * matrix * scale[None, :]
    return torch.from_numpy(np.eye(nodes, dtype=np.float32) - normalized)


class LocalInteractionDynamics(nn.Module):
    """Eq. (4): RMS-normalized MLP for local chemical interactions."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * width, width)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(hidden))


class SpatialTransportDynamics(nn.Module):
    """Eq. (5): normalized-Laplacian pollutant transport."""

    def __init__(self, laplacian: torch.Tensor, width: int) -> None:
        super().__init__()
        self.register_buffer("laplacian", laplacian)
        self.projection = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.einsum("nm,bmd->bnd", self.laplacian, hidden))


class Model(nn.Module):
    """Iterative local/transport/accumulation surrogate forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int | None = None,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.cov_dim = TIME_FEATURES if cov_dim is None else cov_dim
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        self.embed = nn.Linear(1 + self.cov_dim, d_model)
        self.local = LocalInteractionDynamics(d_model, dropout)
        self.transport = SpatialTransportDynamics(_normalized_laplacian(adj_mx, enc_in), d_model)
        self.accumulation = nn.GRUCell(d_model, d_model)
        self.increment = nn.Linear(d_model, 1)
        self.transport_readout = nn.Linear(d_model, 1, bias=False)
        self.last_transport: torch.Tensor | None = None

    def _step(
        self, value: torch.Tensor, covariates: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.embed(torch.cat((value.unsqueeze(-1), covariates), -1))
        hidden = hidden + self.local(hidden)
        message = self.transport(hidden)
        hidden = hidden + message
        b, n, d = hidden.shape
        state = self.accumulation(hidden.reshape(b * n, d), state.reshape(b * n, d)).reshape(b, n, d)
        hidden = hidden + state
        transport = self.transport_readout(message).squeeze(-1)
        # Use only the conservative (zero spatial mean) transport flux in the
        # forecast update; sources/sinks remain represented by the local path.
        conservative_transport = transport - transport.mean(-1, keepdim=True)
        updated = value + self.increment(hidden).squeeze(-1) + conservative_transport
        return updated, state, transport

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        b = x_enc.shape[0]
        history = to_spatiotemporal(x_enc, x_mark_enc)
        history_cov = history[..., 1 : 1 + self.cov_dim]
        if history_cov.shape[-1] < self.cov_dim:
            history_cov = torch.nn.functional.pad(history_cov, (0, self.cov_dim - history_cov.shape[-1]))
        state = x_enc.new_zeros(b, self.enc_in, self.embed.out_features)
        for step in range(self.seq_len):
            _, state, _ = self._step(x_enc[:, step], history_cov[:, step], state)
        marks = x_mark_enc if x_mark_dec is None else x_mark_dec
        if marks is None:
            future_cov = x_enc.new_zeros(b, self.pred_len, self.enc_in, self.cov_dim)
        else:
            future_cov = future_time_features(coerce_time_length(marks, self.pred_len), self.enc_in)
            future_cov = future_cov[..., : self.cov_dim]
            if future_cov.shape[-1] < self.cov_dim:
                future_cov = torch.nn.functional.pad(future_cov, (0, self.cov_dim - future_cov.shape[-1]))
        value = x_enc[:, -1]
        forecasts, transports = [], []
        for step in range(self.pred_len):
            value, state, transport = self._step(value, future_cov[:, step], state)
            forecasts.append(value)
            transports.append(transport)
        self.last_transport = torch.stack(transports, 1)
        return torch.stack(forecasts, 1)

    def domain_informed_constraint(self) -> torch.Tensor:
        """Equations (13)--(16): temporal smoothness and spatial mass balance."""
        if self.last_transport is None:
            raise RuntimeError("run forward before requesting the constraint")
        temporal = self.last_transport[:, 1:] - self.last_transport[:, :-1]
        return temporal.square().mean() + self.last_transport.sum(-1).square().mean()
