"""Clean-room DeepAir following the KDD 2018 distributed-fusion method.

The paper's spatial partition/aggregation/interpolation is represented by an
explicit target-region-source projection.  Its output is the shared main
feature for five residual FusionNets: historical weather, weather forecast,
secondary/spatial pollutant context, meta properties, and holistic influence.
Their horizon-wise outputs use the paper's learnable weighted merge.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.marks import (
    coerce_time_length,
    future_time_features,
    to_spatiotemporal,
)
from tsflab.models._inputs import check_window


def default_spatial_projection(nodes: int, regions: int) -> torch.Tensor:
    """Circular distance-region fallback for datasets without coordinates."""
    regions = min(nodes, regions)
    projection = torch.zeros(nodes, regions, nodes)
    for target in range(nodes):
        for source in range(nodes):
            region = min(regions - 1, ((source - target) % nodes) * regions // nodes)
            distance = min((source - target) % nodes, (target - source) % nodes)
            projection[target, region, source] = 1.0 / (1.0 + distance)
    return projection / projection.sum(-1, keepdim=True).clamp_min(1e-6)


class FusionNet(nn.Module):
    """Concatenation, fully connected interaction, and residual FC block."""

    def __init__(self, inputs: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.input = nn.Linear(inputs, hidden)
        self.residual_1 = nn.Linear(hidden, hidden)
        self.residual_2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, horizon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input(values))
        residual = self.residual_2(torch.relu(self.residual_1(hidden)))
        return self.output(torch.relu(hidden + residual))


class Model(nn.Module):
    """Spatially transformed, heterogeneous distributed-fusion forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        spatial_mx: np.ndarray | torch.Tensor | None = None,
        cov_dim: int = 2,
        hidden_dim: int = 32,
        spatial_regions: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim
        projection = (
            default_spatial_projection(enc_in, spatial_regions)
            if spatial_mx is None
            else torch.as_tensor(spatial_mx, dtype=torch.float32)
        )
        if projection.ndim == 2:
            projection = projection.unsqueeze(1)
        if projection.ndim != 3 or projection.shape[0] != enc_in or projection.shape[2] != enc_in:
            raise ValueError("spatial_mx must have shape (enc_in, regions, enc_in)")
        projection = projection / projection.sum(-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("spatial_projection", projection)
        regions = projection.shape[1]

        self.main_embedding = nn.Linear(seq_len * regions, hidden_dim)
        self.historical_weather_embedding = nn.Linear(seq_len * cov_dim, hidden_dim)
        self.future_weather_embedding = nn.Linear(pred_len * cov_dim, hidden_dim)
        self.spatial_pollutant_embedding = nn.Linear(seq_len * regions, hidden_dim)
        self.station_embedding = nn.Parameter(torch.randn(enc_in, hidden_dim) * 0.02)
        self.calendar_embedding = nn.Linear((seq_len + pred_len) * cov_dim, hidden_dim)

        self.historical_weather = FusionNet(hidden_dim * 2, hidden_dim, pred_len)
        self.weather_forecast = FusionNet(hidden_dim * 2, hidden_dim, pred_len)
        self.secondary_pollutants = FusionNet(hidden_dim * 2, hidden_dim, pred_len)
        self.meta_properties = FusionNet(hidden_dim * 3, hidden_dim, pred_len)
        self.holistic = FusionNet(hidden_dim * 6, hidden_dim, pred_len)
        self.fusion_weights = nn.Parameter(torch.zeros(5, pred_len, enc_in))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], self.seq_len, 6))
        history = to_spatiotemporal(x_enc, x_mark_enc)
        if history.shape[-1] != 1 + self.cov_dim:
            raise ValueError(f"DeepAir expects exactly {self.cov_dim} historical covariates")
        future_marks = x_mark_enc if x_mark_dec is None else x_mark_dec
        future_marks = coerce_time_length(future_marks, self.pred_len)
        future = future_time_features(future_marks, self.enc_in)
        if future.shape[-1] != self.cov_dim:
            raise ValueError(f"DeepAir expects exactly {self.cov_dim} future covariates")

        # Target-relative regional readings: (B, T, target, region).
        transformed = torch.einsum("irn,btn->btir", self.spatial_projection, x_enc)
        flattened = transformed.permute(0, 2, 1, 3).reshape(x_enc.shape[0], self.enc_in, -1)
        main = torch.relu(self.main_embedding(flattened))
        secondary = torch.relu(self.spatial_pollutant_embedding(flattened.square()))
        historical = history[..., 1:].permute(0, 2, 1, 3).reshape(x_enc.shape[0], self.enc_in, -1)
        historical = torch.relu(self.historical_weather_embedding(historical))
        future_flat = future.permute(0, 2, 1, 3).reshape(x_enc.shape[0], self.enc_in, -1)
        forecast = torch.relu(self.future_weather_embedding(future_flat))
        calendar = torch.cat([history[..., 1:].permute(0, 2, 1, 3), future.permute(0, 2, 1, 3)], dim=2).reshape(
            x_enc.shape[0], self.enc_in, -1
        )
        calendar = torch.relu(self.calendar_embedding(calendar))
        station = self.station_embedding.unsqueeze(0).expand(x_enc.shape[0], -1, -1)

        outputs = torch.stack(
            [
                self.historical_weather(torch.cat([main, historical], dim=-1)),
                self.weather_forecast(torch.cat([main, forecast], dim=-1)),
                self.secondary_pollutants(torch.cat([main, secondary], dim=-1)),
                self.meta_properties(torch.cat([main, calendar, station], dim=-1)),
                self.holistic(torch.cat([main, historical, forecast, secondary, calendar, station], dim=-1)),
            ],
            dim=1,
        ).permute(0, 1, 3, 2)
        weights = torch.softmax(self.fusion_weights, dim=0).unsqueeze(0)
        # Paper Eq. (1) assumes min-max-normalized targets and applies sigmoid.
        return torch.sigmoid((outputs * weights).sum(1))
