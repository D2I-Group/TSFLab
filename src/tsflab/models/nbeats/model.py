"""Independent N-BEATS implementation from the neural basis paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def trend_basis(length: int, degree: int, device=None, dtype=None) -> torch.Tensor:
    time = torch.arange(length, device=device, dtype=dtype) / max(length, 1)
    return torch.stack([time.pow(power) for power in range(degree)])


def seasonality_basis(length: int, dimension: int, device=None, dtype=None) -> torch.Tensor:
    time = torch.arange(length, device=device, dtype=dtype) / max(length, 1)
    harmonics = max(1, math.ceil(dimension / 2))
    rows = []
    for frequency in range(harmonics):
        rows.append(torch.cos(2 * math.pi * frequency * time))
        rows.append(torch.sin(2 * math.pi * frequency * time))
    return torch.stack(rows[:dimension])


class NBeatsBlock(nn.Module):
    def __init__(
        self, input_length: int, horizon: int, basis: str, theta_dimension: int, hidden: int, harmonics: int | None
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.horizon = horizon
        self.basis = basis
        dimension = 2 * harmonics if basis == "seasonality" and harmonics else theta_dimension
        layers: list[nn.Module] = [nn.Linear(input_length, hidden), nn.ReLU()]
        for _ in range(3):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.network = nn.Sequential(*layers)
        self.theta_backcast = nn.Linear(hidden, dimension)
        self.theta_forecast = nn.Linear(hidden, dimension)
        if basis == "generic":
            self.backcast_basis = nn.Parameter(torch.empty(dimension, input_length))
            self.forecast_basis = nn.Parameter(torch.empty(dimension, horizon))
            nn.init.xavier_uniform_(self.backcast_basis)
            nn.init.xavier_uniform_(self.forecast_basis)

    def _basis(self, length: int, values: torch.Tensor) -> torch.Tensor:
        if self.basis == "trend":
            return trend_basis(length, self.theta_forecast.out_features, values.device, values.dtype)
        if self.basis == "seasonality":
            return seasonality_basis(length, self.theta_forecast.out_features, values.device, values.dtype)
        return self.backcast_basis if length == self.input_length else self.forecast_basis

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(values)
        backcast = self.theta_backcast(hidden) @ self._basis(self.input_length, values)
        forecast = self.theta_forecast(hidden) @ self._basis(self.horizon, values)
        return backcast, forecast


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        label_len: int,
        features: str,
        enc_in: int,
        stack_types: tuple[str, ...] = ("trend", "seasonality", "generic"),
        nb_blocks_per_stack: int = 3,
        thetas_dim: tuple[int, ...] = (4, 8, 8),
        hidden_layer_units: int = 256,
        share_weights_in_stack: bool = False,
        nb_harmonics: int | None = None,
    ) -> None:
        super().__init__()
        del label_len, features
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        blocks = []
        for basis, dimension in zip(stack_types, thetas_dim):
            shared = NBeatsBlock(seq_len, pred_len, basis, dimension, hidden_layer_units, nb_harmonics)
            blocks.extend(
                [
                    shared
                    if share_weights_in_stack
                    else NBeatsBlock(seq_len, pred_len, basis, dimension, hidden_layer_units, nb_harmonics)
                    for _ in range(nb_blocks_per_stack)
                ]
            )
        self.blocks = nn.ModuleList(blocks)
        if not share_weights_in_stack:
            for parameter in self.blocks[-1].theta_backcast.parameters():
                parameter.requires_grad_(False)
            if self.blocks[-1].basis == "generic":
                self.blocks[-1].backcast_basis.requires_grad_(False)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        batch, _, channels = x_enc.shape
        residual = x_enc.transpose(1, 2).reshape(batch * channels, self.seq_len)
        forecast = residual.new_zeros(batch * channels, self.pred_len)
        for index, block in enumerate(self.blocks):
            backcast, partial = block(residual)
            if index + 1 < len(self.blocks):
                residual = residual - backcast
            forecast = forecast + partial
        return forecast.reshape(batch, channels, self.pred_len).transpose(1, 2)
