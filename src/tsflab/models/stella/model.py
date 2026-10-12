"""Local STELLA implementation (paper Eqs. 3-6).

Independent rewrite from the paper; the pinned official repository (no
license file) was read only to resolve omissions such as the time step that
indexes the temporal embedding and the residual MLP layout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.marks import TIME_FEATURES
from tsflab.models._inputs import check_window

#: Sizes of the hour-of-day, day-of-month and month-of-year tables (Eq. 4).
HOURS, DAYS, MONTHS = 24, 31, 12

# Columns of the raw ``[year, month, day, weekday, hour, minute]`` marks.
_MONTH, _DAY, _HOUR = 1, 2, 4


def standardize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Standardize each coordinate column independently (population std).

    A constant column keeps unit scale, so it maps to zero instead of NaN.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("station coordinates must have shape [nodes, 3]")
    if not np.isfinite(values).all():
        raise ValueError("station coordinates must be finite")
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return ((values - values.mean(axis=0)) / scale).astype(np.float32)


class SpatialEmbedding(nn.Module):
    """Eq. (3) ``SE_i = W2 ReLU(W1 Sigma_i + b1) + b2`` over standardized coordinates.

    Without coordinates it falls back to one learnable vector per station,
    the paper's relative-position (RPE) ablation.
    """

    def __init__(self, num_nodes: int, d_model: int, coordinates: np.ndarray | None) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        if coordinates is None:
            self.uses_coordinates = False
            self.table = nn.Embedding(num_nodes, d_model)
        else:
            positions = standardize_coordinates(coordinates)
            if positions.shape[0] != num_nodes:
                raise ValueError(
                    f"expected coordinates for {num_nodes} stations, got {positions.shape[0]}"
                )
            self.uses_coordinates = True
            self.register_buffer("positions", torch.from_numpy(positions))
            self.ffn = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self) -> torch.Tensor:
        """Station embeddings ``[nodes, d_model]``."""
        if self.uses_coordinates:
            return self.ffn(self.positions)
        return self.table.weight


class TemporalEmbedding(nn.Module):
    """Eq. (4) ``TE_t = T_hour + D_day + M_month`` with learnable tables."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.hour = nn.Embedding(HOURS, d_model)
        self.day = nn.Embedding(DAYS, d_model)
        self.month = nn.Embedding(MONTHS, d_model)

    @staticmethod
    def indices(marks: torch.Tensor | None, batch: int, device) -> dict[str, torch.Tensor]:
        """Zero-based calendar indices of the first history step.

        Raw ``[B, T, 6]`` stamps give hour, day of month and month. Node
        calendar covariates ``[B, T, N, 2]`` (``[time_in_day, day_in_week]``)
        only give the hour, so the day and month terms are omitted.
        """
        if marks is None:
            zeros = torch.zeros(batch, dtype=torch.long, device=device)
            return {"hour": zeros, "day": zeros, "month": zeros}
        if marks.ndim == 3:
            first = marks[:, 0]
            return {
                "hour": first[:, _HOUR].long().clamp(0, HOURS - 1),
                "day": (first[:, _DAY].long() - 1).clamp(0, DAYS - 1),
                "month": (first[:, _MONTH].long() - 1).clamp(0, MONTHS - 1),
            }
        if marks.ndim == 4 and marks.shape[-1] == TIME_FEATURES:
            hour = torch.floor(marks[:, 0, 0, 0] * HOURS).long().clamp(0, HOURS - 1)
            return {"hour": hour}
        raise ValueError(
            "STELLA needs raw [batch, time, 6] calendar marks or node calendar "
            f"covariates [batch, time, nodes, {TIME_FEATURES}]"
        )

    def forward(self, indices: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample temporal embedding ``[batch, d_model]``."""
        tables = {"hour": self.hour, "day": self.day, "month": self.month}
        return sum(tables[name](index) for name, index in indices.items())


class ResidualMLP(nn.Module):
    """Eq. (6) ``Z_{l+1} = FFN_l(Z_l) + Z_l`` with Linear-ReLU-Linear-Dropout."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.ffn(z) + z


class Model(nn.Module):
    """STELLA: spatial-temporal position embedding plus a residual MLP encoder."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        coordinates: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.input_embedding = nn.Linear(seq_len, d_model)
        self.spatial = SpatialEmbedding(enc_in, d_model, coordinates)
        self.temporal = TemporalEmbedding(d_model)
        self.encoder = nn.Sequential(*[ResidualMLP(d_model, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, pred_len)

    def embed(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """Eq. (5) ``E = Linear(X_i) + SE_i + TE_t``: ``[batch, nodes, d_model]``."""
        hidden = self.input_embedding(x_enc.transpose(1, 2))
        indices = self.temporal.indices(x_mark_enc, x_enc.shape[0], x_enc.device)
        return hidden + self.spatial().unsqueeze(0) + self.temporal(indices).unsqueeze(1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        encoded = self.encoder(self.embed(x_enc, x_mark_enc))
        return self.output_layer(encoded).transpose(1, 2).contiguous()


__all__ = [
    "Model",
    "ResidualMLP",
    "SpatialEmbedding",
    "TemporalEmbedding",
    "standardize_coordinates",
]
