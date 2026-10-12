"""Clean-room RPMixer implementation from the KDD 2024 paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._inputs import check_window


class ComplexTemporalProjection(nn.Module):
    """Paper Equation (1): a learned complex map in the FFT domain."""

    def __init__(self, length: int) -> None:
        super().__init__()
        scale = length**-0.5
        self.real_weight = nn.Parameter(torch.randn(length, length) * scale)
        self.imag_weight = nn.Parameter(torch.randn(length, length) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(x, dim=-1)
        weight = torch.complex(self.real_weight, self.imag_weight)
        transformed = torch.einsum("oi,bni->bno", weight, spectrum)
        return torch.fft.ifft(transformed, dim=-1).real


class FixedRandomProjection(nn.Module):
    """A reproducible, non-trainable Johnson--Lindenstrauss projection."""

    def __init__(self, in_features: int, out_features: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(out_features, in_features, generator=generator)
        self.register_buffer("weight", weight / math.sqrt(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RPMixerBlock(nn.Module):
    """Equations (2--6): pre-activation temporal/spatial residual paths."""

    def __init__(self, seq_len: int, nodes: int, random_dim: int, seed: int) -> None:
        super().__init__()
        self.temporal = ComplexTemporalProjection(seq_len)
        self.random_projection = FixedRandomProjection(nodes, random_dim, seed)
        self.spatial_reconstruction = nn.Linear(random_dim, nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = x + self.temporal(F.relu(x))
        across_nodes = temporal.transpose(1, 2)
        random_view = self.random_projection(F.relu(across_nodes))
        spatial_delta = self.spatial_reconstruction(F.relu(random_view))
        return temporal + spatial_delta.transpose(1, 2)


class Model(nn.Module):
    """Graph-free spatial-temporal RPMixer with a shared horizon decoder."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        random_dim: int = 4,
        e_layers: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.blocks = nn.ModuleList(
            RPMixerBlock(seq_len, enc_in, random_dim, 104729 + index)
            for index in range(e_layers)
        )
        self.decoder = ChannelWiseLinear(seq_len, pred_len, enc_in)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        hidden = x_enc.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.decoder(hidden).transpose(1, 2)
