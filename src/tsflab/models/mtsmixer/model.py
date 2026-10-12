"""Clean-room MTS-Mixers implementation from the paper equations."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.revin import RevIN


class TemporalSubsequenceMixer(nn.Module):
    """Equation (6): learn interleaved subsequences independently and merge."""

    def __init__(
        self, seq_len: int, sampling: int, hidden_dim: int, factorized: bool
    ) -> None:
        super().__init__()
        groups = sampling if factorized else 1
        if groups > seq_len:
            raise ValueError("sampling cannot exceed seq_len")
        self.groups = groups
        self.paths = nn.ModuleList()
        for offset in range(groups):
            length = len(range(offset, seq_len, groups))
            self.paths.append(
                nn.Sequential(
                    nn.Linear(length, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, length),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.zeros_like(x)
        for offset, path in enumerate(self.paths):
            mixed[:, :, offset :: self.groups] = path(x[:, :, offset :: self.groups])
        return mixed


class ChannelInteraction(nn.Module):
    """Equation (8): a channel bottleneck U,V with a nonlinearity."""

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, hidden_dim)
        self.expand = nn.Linear(hidden_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(torch.nn.functional.gelu(self.reduce(x)))


class FactorizedMixerBlock(nn.Module):
    """Paper Equation (3), with optional factorization ablations."""

    def __init__(
        self,
        seq_len: int,
        channels: int,
        temporal_hidden: int,
        channel_hidden: int,
        sampling: int,
        factorized_temporal: bool,
        factorized_channel: bool,
        normalize: bool,
    ) -> None:
        super().__init__()
        self.pre_time = nn.LayerNorm(channels) if normalize else nn.Identity()
        self.pre_channel = nn.LayerNorm(channels) if normalize else nn.Identity()
        self.temporal = TemporalSubsequenceMixer(
            seq_len, sampling, temporal_hidden, factorized_temporal
        )
        channel_width = channel_hidden if factorized_channel else temporal_hidden
        self.channel = ChannelInteraction(channels, channel_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_delta = self.temporal(self.pre_time(x).transpose(1, 2)).transpose(1, 2)
        joined = x + temporal_delta
        return joined + self.channel(self.pre_channel(joined))


class Model(nn.Module):
    """Factorized temporal/channel mixing followed by a linear forecast map."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        d_model: int = 64,
        d_ff: int = 4,
        e_layers: int = 2,
        fac_T: bool = True,
        fac_C: bool = True,
        sampling: int = 2,
        norm: bool = True,
        individual: bool = False,
        rev: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) <= 0:
            raise ValueError("all dimensions, layers, and sampling must be positive")
        if features not in {"M", "MS", "S"}:
            raise ValueError("features must be one of M, MS, or S")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.normalization = RevIN(enc_in, affine=False) if rev else None
        self.blocks = nn.ModuleList(
            FactorizedMixerBlock(
                seq_len,
                enc_in,
                d_model,
                d_ff,
                sampling,
                fac_T,
                fac_C,
                norm,
            )
            for _ in range(e_layers)
        )
        self.final_norm = nn.LayerNorm(enc_in) if norm else nn.Identity()
        self.projection = ChannelWiseLinear(seq_len, pred_len, enc_in, individual)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("MTSMixer expects (batch, configured seq_len, enc_in)")
        x = self.normalization(x_enc, "norm") if self.normalization else x_enc
        for block in self.blocks:
            x = block(x)
        forecast = self.projection(self.final_norm(x).transpose(1, 2)).transpose(1, 2)
        if self.normalization:
            forecast = self.normalization(forecast, "denorm")
        if self.features == "MS":
            return forecast[:, :, -1:]
        return forecast
