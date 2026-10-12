"""DecompRevisit: dual-MLP forecaster over a moving-average trend/seasonal split.

Independent implementation of the RMSM and RMM models of Section 2 (Eqs. 1-2,
8, 10-14) of Panta et al., "Revisiting the Seasonal Trend Decomposition for
Enhanced Time Series Forecasting" (arXiv 2602.18465, ICASSP 2026), after reading
the pinned official code (``Sanjeev97/Time-Series-Decomposition`` at
``73f0a4b4``, no license file) to resolve omissions; nothing is copied.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import SeriesDecomposition


def three_layer_mlp(seq_len: int, hidden: int, pred_len: int) -> nn.Sequential:
    """Eqs. (8) and (13): Linear-ReLU-Linear-ReLU-Linear over the time axis."""
    return nn.Sequential(
        nn.Linear(seq_len, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, pred_len),
    )


class ShiftMLP(nn.Module):
    """Eqs. (10)-(12), shift forecasting of the seasonal part on ``[B, C, L]``:
    ``h1 = ReLU(W1 X^S)``, ``h2 = ReLU(W2 [h1, X])`` with the raw input ``X`` as
    context, ``Y^S = W3 h2`` (linear output)."""

    def __init__(self, seq_len: int, pred_len: int, hidden: int = 64, mix_hidden: int = 128) -> None:
        super().__init__()
        self.seasonal = nn.Linear(seq_len, hidden)
        self.mix = nn.Linear(hidden + seq_len, mix_hidden)
        self.out = nn.Linear(mix_hidden, pred_len)

    def forward(self, seasonal: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.seasonal(seasonal))
        h2 = torch.relu(self.mix(torch.cat((h1, raw), dim=-1)))
        return self.out(h2)


class Model(nn.Module):
    """RMSM (``seasonal_head="shift_mlp"``) or RMM (``"mlp"``) on ``[B, L, C]``."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_size: int = 25,
        trend_hidden: int = 512,
        seasonal_head: Literal["shift_mlp", "mlp"] = "shift_mlp",
        shift_hidden: int = 64,
        shift_mix_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seasonal_head = seasonal_head
        self.decomposition = SeriesDecomposition(kernel_size)
        self.revin_trend = RevIN(enc_in)
        self.trend_mlp = three_layer_mlp(seq_len, trend_hidden, pred_len)
        # RMM reuses the trend MLP for the seasonal part (official code); RMSM has its own head.
        self.shift_mlp = (
            ShiftMLP(seq_len, pred_len, shift_hidden, shift_mix_hidden) if seasonal_head == "shift_mlp" else None
        )

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eqs. (1)-(2): ``X^T = AvgPool(Padding(X))``, ``X^S = X - X^T``; returns
        ``(seasonal, trend)``."""
        return self.decomposition(x)

    def forecast_trend(self, trend: torch.Tensor) -> torch.Tensor:
        """Eq. (8): ``Y^T = RevIN^-1(MLP(RevIN(X^T)))``."""
        normalized = self.revin_trend(trend, "norm")
        out = self.trend_mlp(normalized.transpose(1, 2)).transpose(1, 2)
        return self.revin_trend(out, "denorm")

    def forecast_seasonal(self, seasonal: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Seasonal part without any normalization: Eqs. (10)-(12) or Eq. (13)."""
        if self.shift_mlp is not None:
            out = self.shift_mlp(seasonal.transpose(1, 2), raw.transpose(1, 2))
        else:
            out = self.trend_mlp(seasonal.transpose(1, 2))
        return out.transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        seasonal, trend = self.decompose(x_enc)
        # Eq. (14): Y = Y^T + Y^S.
        return self.forecast_trend(trend) + self.forecast_seasonal(seasonal, x_enc)
