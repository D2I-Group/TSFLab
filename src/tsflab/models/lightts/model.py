"""Clean-room LightTS implementation following paper Equations 1--2 and IEBlock."""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._inputs import check_window


class InformationExchangeBlock(nn.Module):
    """Bottleneck temporal/channel/output projections from Section 3.4."""

    def __init__(
        self,
        rows: int,
        channels: int,
        bottleneck: int,
        output_rows: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_projection = nn.Sequential(nn.Linear(rows, bottleneck), nn.LeakyReLU(), nn.Dropout(dropout))
        self.channel_projection = nn.Linear(channels, channels)
        self.output_projection = nn.Linear(bottleneck, output_rows)
        nn.init.eye_(self.channel_projection.weight)
        nn.init.zeros_(self.channel_projection.bias)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        # Matrix convention is [batch, H, W], matching the paper.
        temporal = self.temporal_projection(matrix.transpose(1, 2)).transpose(1, 2)
        exchanged = temporal + self.channel_projection(temporal)
        return self.output_projection(exchanged.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hid_dim: int = 128,
        dropout: float = 0.0,
        chunk_size: int = 24,
    ) -> None:
        super().__init__()
        if seq_len % chunk_size:
            raise ValueError("seq_len must be divisible by chunk_size")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.chunk_size = chunk_size
        self.num_chunks = seq_len // chunk_size
        feature_width = hid_dim // 4
        bottleneck = max(1, feature_width // 4)
        self.continuous_block = InformationExchangeBlock(
            chunk_size, self.num_chunks, bottleneck, feature_width, dropout
        )
        self.interval_block = InformationExchangeBlock(chunk_size, self.num_chunks, bottleneck, feature_width, dropout)
        self.continuous_summary = nn.Linear(self.num_chunks, 1)
        self.interval_summary = nn.Linear(self.num_chunks, 1)
        self.forecast_block = InformationExchangeBlock(
            2 * feature_width, enc_in, max(1, hid_dim // 8), pred_len, dropout
        )
        self.highway = nn.Linear(seq_len, pred_len)

    def sample_continuous(self, series: torch.Tensor) -> torch.Tensor:
        """Equation 1: columns are consecutive, non-overlapping subsequences."""
        return series.reshape(-1, self.num_chunks, self.chunk_size).transpose(1, 2)

    def sample_interval(self, series: torch.Tensor) -> torch.Tensor:
        """Equation 2: each column uses stride floor(T/C)=T/C."""
        return series.reshape(-1, self.chunk_size, self.num_chunks)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        per_series = x_enc.transpose(1, 2).reshape(batch * self.enc_in, self.seq_len)
        continuous = self.continuous_summary(self.continuous_block(self.sample_continuous(per_series))).squeeze(-1)
        interval = self.interval_summary(self.interval_block(self.sample_interval(per_series))).squeeze(-1)
        features = torch.cat((continuous, interval), dim=-1)
        features = features.view(batch, self.enc_in, -1).transpose(1, 2)
        nonlinear = self.forecast_block(features)
        linear = self.highway(x_enc.transpose(1, 2)).transpose(1, 2)
        return nonlinear + linear
