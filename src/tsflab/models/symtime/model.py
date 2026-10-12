"""Clean-room SymTime forecasting path based on the published method.

SymTime's downstream reconstruction path uses instance normalization, explicit
trend/periodic decomposition, the pre-training paper's non-overlapping patch
Transformer as the periodic-series encoder, and a direct trend regressor. The
symbol encoder and momentum pre-training machinery are intentionally outside
the forecasting-only runtime contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._inputs import check_window


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        patch_len: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        trend_kernel: int = 25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.num_patches = math.ceil(seq_len / patch_len)

        self.revin = RevIN(enc_in)
        self.decomposition = SeriesDecomposition(trend_kernel)
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.series_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.periodic_head = nn.Linear(self.num_patches * d_model, pred_len)
        self.trend_head = nn.Linear(seq_len, pred_len)

    def patch_series(self, periodic: torch.Tensor) -> torch.Tensor:
        """Create the paper's non-overlapping time-series patches."""
        values = periodic.transpose(1, 2)
        padded_length = self.num_patches * self.patch_len
        if padded_length > self.seq_len:
            values = F.pad(values, (0, padded_length - self.seq_len), mode="replicate")
        return values.unfold(-1, self.patch_len, self.patch_len)

    def encode_periodic(self, periodic: torch.Tensor) -> torch.Tensor:
        patches = self.patch_series(periodic)
        batch, channels, count, width = patches.shape
        tokens = self.patch_projection(patches.reshape(batch * channels, count, width))
        return self.series_encoder(tokens + self.position).reshape(
            batch, channels, count, -1
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        periodic, trend = self.decomposition(normalized)
        encoded = self.encode_periodic(periodic)
        periodic_forecast = self.periodic_head(encoded.flatten(-2)).transpose(1, 2)
        trend_forecast = self.trend_head(trend.transpose(1, 2)).transpose(1, 2)
        return self.revin(periodic_forecast + trend_forecast, "denorm")


__all__ = ["Model"]
