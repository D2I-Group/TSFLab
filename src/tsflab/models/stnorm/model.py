"""Local ST-Norm implementation from paper and reference-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._inputs import check_window


class SpatialNormalization(nn.Module):
    """Normalize each time step over the node axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance, mean = torch.var_mean(x, dim=2, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.scale + self.shift


class TemporalNormalization(nn.Module):
    """Normalize each node over batch and time with running evaluation stats."""

    def __init__(self, nodes: int, channels: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, nodes, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, nodes, 1))
        self.register_buffer("running_mean", torch.zeros(1, channels, nodes, 1))
        self.register_buffer("running_var", torch.ones(1, channels, nodes, 1))
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            variance, mean = torch.var_mean(x, dim=(0, 3), keepdim=True, unbiased=False)
            self.running_mean.lerp_(mean.detach(), self.momentum)
            self.running_var.lerp_(variance.detach(), self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.scale + self.shift


class NormalizedTemporalLayer(nn.Module):
    """Concatenate raw, spatial-normalized and temporal-normalized streams."""

    def __init__(self, nodes: int, channels: int, kernel: int, dilation: int, use_spatial: bool, use_temporal: bool) -> None:
        super().__init__()
        self.spatial = SpatialNormalization(channels) if use_spatial else None
        self.temporal = TemporalNormalization(nodes, channels) if use_temporal else None
        streams = 1 + int(use_spatial) + int(use_temporal)
        self.padding = dilation * (kernel - 1)
        self.filter = nn.Conv2d(streams * channels, channels, (1, kernel), dilation=(1, dilation))
        self.gate = nn.Conv2d(streams * channels, channels, (1, kernel), dilation=(1, dilation))
        self.residual = nn.Conv2d(channels, channels, 1)
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streams = [x]
        if self.spatial is not None:
            streams.append(self.spatial(x))
        if self.temporal is not None:
            streams.append(self.temporal(x))
        combined = F.pad(torch.cat(streams, 1), (self.padding, 0, 0, 0))
        gated = torch.tanh(self.filter(combined)) * torch.sigmoid(self.gate(combined))
        hidden = x + self.residual(gated)
        return hidden, self.skip(hidden)


class Model(nn.Module):
    """WaveNet temporal backbone augmented with spatial and temporal normalization."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, input_dim: int = 3, channels: int = 16, kernel_size: int = 2, blocks: int = 2, layers: int = 2, tnorm_bool: bool = True, snorm_bool: bool = True) -> None:
        super().__init__()
        del adj_mx
        self.seq_len, self.pred_len, self.num_nodes, self.input_dim = seq_len, pred_len, num_nodes, input_dim
        self.input_projection = nn.Conv2d(input_dim, channels, 1)
        self.layers = nn.ModuleList(
            NormalizedTemporalLayer(num_nodes, channels, kernel_size, 2**layer, snorm_bool, tnorm_bool)
            for _ in range(blocks)
            for layer in range(layers)
        )
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 1), nn.ReLU(), nn.Conv2d(channels, pred_len, 1))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        check_window(x_enc, self.seq_len, self.num_nodes)
        data = to_spatiotemporal(x_enc, x_mark_enc)
        if data.shape[-1] < self.input_dim:
            raise ValueError("STNorm received fewer input features than configured")
        hidden = self.input_projection(data[..., : self.input_dim].permute(0, 3, 2, 1))
        skips = None
        for layer in self.layers:
            hidden, skip = layer(hidden)
            skips = skip if skips is None else skips + skip
        assert skips is not None
        return self.output(skips)[..., -1]


__all__ = ["Model", "SpatialNormalization", "TemporalNormalization", "NormalizedTemporalLayer"]
