"""Independent N-HiTS implementation with multi-rate hierarchical interpolation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


class NHiTSBlock(nn.Module):
    def __init__(self, input_length: int, horizon: int, pool_size: int,
                 downsample: int, units: list[int], pooling: str,
                 interpolation: str, dropout: float, activation: str) -> None:
        super().__init__()
        self.input_length = input_length
        self.horizon = horizon
        self.pool_size = pool_size
        self.interpolation = interpolation
        pooled_length = math.ceil(input_length / pool_size)
        self.coefficient_count = math.ceil(horizon / downsample)
        dimensions = [pooled_length, *units]
        layers: list[nn.Module] = []
        activation_cls = getattr(nn, activation)
        for source, target in zip(dimensions, dimensions[1:]):
            layers.extend([nn.Linear(source, target), activation_cls(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)
        final = dimensions[-1]
        # The hierarchical constraint applies to forecast knots only.  The
        # backcast spans the complete input window so the next block receives
        # an unrestricted residual at its own frequency scale.
        self.backcast_coefficients = nn.Linear(final, input_length)
        self.forecast_coefficients = nn.Linear(final, self.coefficient_count)
        self.pooling = pooling

    def _interpolate(self, coefficients: torch.Tensor, length: int) -> torch.Tensor:
        mode = self.interpolation.lower()
        if mode in {"nearest", "nearest-exact"}:
            return F.interpolate(coefficients.unsqueeze(1), size=length, mode=mode).squeeze(1)
        return F.interpolate(coefficients.unsqueeze(1), size=length, mode=mode,
                             align_corners=False).squeeze(1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = (-values.shape[-1]) % self.pool_size
        padded = F.pad(values.unsqueeze(1), (pad, 0), mode="replicate")
        pool = F.max_pool1d if self.pooling == "MaxPool1d" else F.avg_pool1d
        pooled = pool(padded, self.pool_size, stride=self.pool_size).squeeze(1)
        hidden = self.network(pooled)
        backcast = self.backcast_coefficients(hidden)
        forecast = self._interpolate(self.forecast_coefficients(hidden), self.horizon)
        return backcast, forecast


class Model(nn.Module):
    def __init__(
        self, seq_len: int, pred_len: int, label_len: int, features: str,
        enc_in: int, stack_types: list[str], n_blocks: list[int], mlp_units: list,
        n_pool_kernel_size: list[int], n_freq_downsample: list[int],
        pooling_mode: str = "MaxPool1d", interpolation_mode: str = "linear",
        dropout: float = 0.0, activation: str = "ReLU", use_norm: bool = True,
    ) -> None:
        super().__init__()
        del label_len, features
        count = len(stack_types)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.normalization = RevIN(enc_in, affine=False, enabled=use_norm)
        blocks = []
        for index in range(count):
            units = mlp_units[index] if len(mlp_units) == count else mlp_units[0]
            for _ in range(n_blocks[index]):
                blocks.append(NHiTSBlock(
                    seq_len, pred_len, n_pool_kernel_size[index],
                    n_freq_downsample[index], list(units), pooling_mode,
                    interpolation_mode, dropout, activation,
                ))
        self.blocks = nn.ModuleList(blocks)
        for parameter in self.blocks[-1].backcast_coefficients.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        normalized = self.normalization(x_enc, "norm")
        batch, _, channels = normalized.shape
        residual = normalized.transpose(1, 2).reshape(batch * channels, self.seq_len)
        # Initialize with the last observed level before accumulating block
        # forecasts, as in the paper/reference residual stack.
        forecast = residual[:, -1:].expand(-1, self.pred_len).clone()
        for index, block in enumerate(self.blocks):
            backcast, partial = block(residual)
            if index + 1 < len(self.blocks):
                residual = residual - backcast
            forecast = forecast + partial
        prediction = forecast.reshape(batch, channels, self.pred_len).transpose(1, 2)
        return self.normalization(prediction, "denorm")
