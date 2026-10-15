"""Clean-room CrossLinear implementation from the published equations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class CrossCorrelationEmbedding(nn.Module):
    """Equation (7--8): one direct, time-invariant cross-variate map."""

    def __init__(self, channels: int, alpha: float) -> None:
        super().__init__()
        self.direct_map = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * self.direct_map(x)


class PatchForecastHead(nn.Module):
    """Equations (9--11): patch projection, position blend, global head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        beta: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_count = math.ceil(seq_len / patch_len)
        self.patch_projection = nn.Sequential(
            nn.Linear(patch_len, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = nn.Parameter(torch.randn(1, 1, self.patch_count, d_model) * 0.02)
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.forecast = nn.Linear(self.patch_count * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.patch_count * self.patch_len - x.shape[-1]
        patches = F.pad(x, (0, padding)).unfold(-1, self.patch_len, self.patch_len)
        values = self.patch_projection(patches)
        embedded = self.beta * values + (1.0 - self.beta) * self.position
        return self.forecast(embedded.flatten(start_dim=2))


class Model(nn.Module):
    """Many-to-many CrossLinear with weight sharing across target variables."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.normalization = RevIN(enc_in, affine=False)
        self.cross_embedding = CrossCorrelationEmbedding(enc_in, alpha)
        self.head = PatchForecastHead(seq_len, pred_len, patch_len, d_model, d_ff, beta)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.normalization(x_enc, "norm").transpose(1, 2)
        forecast = self.head(self.cross_embedding(normalized)).transpose(1, 2)
        return self.normalization(forecast, "denorm")
