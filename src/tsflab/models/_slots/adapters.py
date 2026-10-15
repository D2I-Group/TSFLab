"""Torch slot adapters wrapping cataloged components with one uniform interface.

Every adapter respects the interface in its component card: shapes, axis order,
and constructor arguments are used exactly as documented there. ``Pipeline``
chains the six slots (loss lives in the run config, not here)::

    x [B, L, C] -> normalization.norm -> decomposition.split -> per-part temporal
    encoders (z [B, C, D, P]) -> concat along P -> head -> normalization.denorm
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.gated_dilated_conv import gated_dilated_conv
from tsflab.models._components.gaussian_parameter_head import GaussianParameterHead
from tsflab.models._components.last_value_center import center_on_last_value, restore_last_value
from tsflab.models._components.mamba import MambaResidualBlock
from tsflab.models._components.mixer_block import MixerBlock
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.quantile_head import QuantileHead, validate_quantile_levels
from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._components.tst_transformer import TSTEncoder
from tsflab.models._slots.registry import OPTIONS, check_assignment

# --------------------------------------------------------------- normalization


class NoNorm(nn.Module):
    def norm(self, x):
        return x

    def denorm(self, y):
        return y

    def rescale(self, scale):
        return scale


class RevINNorm(nn.Module):
    """RevIN(C, affine=False): a pure positive affine map, so quantile order survives."""

    def __init__(self, c_in: int) -> None:
        super().__init__()
        self.revin = RevIN(c_in, affine=False)

    def norm(self, x):
        return self.revin(x, "norm")

    def denorm(self, y):
        if y.ndim == 3:
            return self.revin(y, "denorm")
        # [B, H, C, Q]: the card's statistics are [B, 1, C]; denormalize each trailing slice.
        return torch.stack([self.revin(y[..., q], "denorm") for q in range(y.shape[-1])], dim=-1)

    def rescale(self, scale):
        return scale * self.revin._scale


class LastValueNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._level = None

    def norm(self, x):
        centered, self._level = center_on_last_value(x)
        return centered

    def denorm(self, y):
        level = self._level if y.ndim == 3 else self._level.unsqueeze(-1)
        return restore_last_value(y, level)

    def rescale(self, scale):
        return scale  # centering only; no scale change


def build_normalization(name: str, c_in: int) -> nn.Module:
    return {"none": lambda: NoNorm(), "revin": lambda: RevINNorm(c_in), "last_value_center": lambda: LastValueNorm()}[
        name
    ]()


# --------------------------------------------------------------- decomposition


class NoDecomposition(nn.Module):
    branches = 1

    def split(self, x):
        return [x]


class SeriesDecompositionSplit(nn.Module):
    branches = 2

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.decomposition = SeriesDecomposition(kernel_size)

    def split(self, x):
        residual, trend = self.decomposition(x)
        return [residual, trend]


def build_decomposition(name: str, kernel_size: int) -> nn.Module:
    return NoDecomposition() if name == "none" else SeriesDecompositionSplit(kernel_size)


# -------------------------------------------------------------------- temporal


class _TemporalBase(nn.Module):
    """``encode(part [B, L, C]) -> z [B, C, D, P]``; ``out_shape`` is ``(D, P)``."""

    out_shape: tuple[int, int]


class LinearHistory(_TemporalBase):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.out_shape = (1, seq_len)

    def encode(self, x):
        return x.transpose(1, 2).unsqueeze(2)


class ChannelWiseLinearTemporal(_TemporalBase):
    def __init__(self, seq_len: int, hidden: int, c_in: int, individual: bool) -> None:
        super().__init__()
        self.projection = ChannelWiseLinear(seq_len, hidden, c_in, individual)
        self.out_shape = (1, hidden)

    def encode(self, x):
        return self.projection(x.transpose(1, 2)).unsqueeze(2)


class _PatchTokens(_TemporalBase):
    """Patchify each channel independently: ``[B, L, C] -> [B*C, P, patch_len]``."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        if not 1 <= patch_len <= seq_len or stride < 1:
            raise ValueError("patch_len must be in [1, seq_len] and stride >= 1")
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.projection = nn.Linear(patch_len, d_model)
        self.out_shape = (d_model, self.n_patches)

    def tokens(self, x):
        batch, _, channels = x.shape
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)  # [B, C, P, patch_len]
        return self.projection(patches.reshape(batch * channels, self.n_patches, self.patch_len)), batch, channels

    @staticmethod
    def to_z(h, batch, channels):
        return h.reshape(batch, channels, h.shape[1], h.shape[2]).transpose(2, 3)  # [B, C, d_model, P]


class TSTTemporal(_PatchTokens):
    def __init__(self, seq_len, patch_len, stride, d_model, n_heads, n_layers, dropout) -> None:
        super().__init__(seq_len, patch_len, stride, d_model)
        self.position = positional_encoding("sincos", False, self.n_patches, d_model)
        self.encoder = TSTEncoder(
            d_model,
            n_heads,
            n_layers=n_layers,
            d_ff=2 * d_model,
            attn_dropout=dropout,
            res_dropout=dropout,
            ffn_dropout=dropout,
            proj_dropout=dropout,
        )

    def encode(self, x):
        h, batch, channels = self.tokens(x)
        return self.to_z(self.encoder(h + self.position), batch, channels)


class MambaTemporal(_PatchTokens):
    def __init__(self, seq_len, patch_len, stride, d_model, n_layers) -> None:
        super().__init__(seq_len, patch_len, stride, d_model)
        self.blocks = nn.ModuleList(
            MambaResidualBlock(d_model, 2 * d_model, max(1, d_model // 16), 4, 8) for _ in range(n_layers)
        )

    def encode(self, x):
        h, batch, channels = self.tokens(x)
        for block in self.blocks:
            h = block(h)
        return self.to_z(h, batch, channels)


class GatedConvTemporal(_TemporalBase):
    def __init__(self, seq_len: int, hidden: int, n_layers: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.lift = nn.Conv1d(1, hidden, 1)
        self.filters = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel_size, dilation=2**i) for i in range(n_layers))
        self.gates = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel_size, dilation=2**i) for i in range(n_layers))
        self.out_shape = (hidden, seq_len)

    def encode(self, x):
        batch, length, channels = x.shape
        h = self.lift(x.transpose(1, 2).reshape(batch * channels, 1, length))
        for filt, gate in zip(self.filters, self.gates):
            h = h + gated_dilated_conv(h, filt, gate)
        return h.reshape(batch, channels, h.shape[1], length)


class MixerTemporal(_TemporalBase):
    def __init__(self, seq_len, c_in, hidden, n_layers, dropout) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(MixerBlock(seq_len, c_in, hidden, dropout) for _ in range(n_layers))
        self.out_shape = (1, seq_len)

    def encode(self, x):
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2).unsqueeze(2)


# ------------------------------------------------------------------------ head


class _HeadBase(nn.Module):
    output_type = "point"

    def project(self, z, normalization):  # pragma: no cover - interface
        raise NotImplementedError


class FlattenHead(_HeadBase):
    def __init__(self, individual: bool, c_in: int, nf: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        self.head = FlattenForecastHead(individual, c_in, nf, pred_len, dropout)

    def base(self, z):
        return self.head(z).transpose(1, 2)  # [B, H, C]

    def project(self, z, normalization):
        return normalization.denorm(self.base(z))


class QuantileProjection(FlattenHead):
    output_type = "quantile"

    def __init__(self, individual, c_in, nf, pred_len, dropout, quantile_levels) -> None:
        super().__init__(individual, c_in, nf, pred_len, dropout)
        self.quantiles = QuantileHead(validate_quantile_levels(quantile_levels), in_features=1)

    def project(self, z, normalization):
        return normalization.denorm(self.quantiles(self.base(z).unsqueeze(-1)))  # [B, H, C, Q]


class GaussianProjection(_HeadBase):
    output_type = "distribution"

    def __init__(self, c_in: int, nf: int, pred_len: int) -> None:
        super().__init__()
        self.head = GaussianParameterHead(nf, pred_len)

    def project(self, z, normalization):
        loc, scale = self.head(z.flatten(2))  # [B, C, H] each
        loc = normalization.denorm(loc.transpose(1, 2))
        scale = normalization.rescale(scale.transpose(1, 2))
        return torch.stack((loc, scale), dim=-1)  # [B, H, C, 2]


# -------------------------------------------------------------------- pipeline


class Pipeline(nn.Module):
    """Six-slot forecaster; ``assignment`` maps slot -> canonical option name."""

    def __init__(
        self,
        assignment: dict[str, str],
        c_in: int,
        seq_len: int,
        pred_len: int,
        *,
        hidden: int = 16,
        n_layers: int = 1,
        n_heads: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        kernel_size: int = 25,
        dropout: float = 0.0,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        problems = check_assignment(assignment)
        if problems:
            raise ValueError("; ".join(problems))
        if min(c_in, seq_len, pred_len, hidden, n_layers) <= 0:
            raise ValueError("channels, lengths, hidden width, and layers must be positive")
        self.assignment = dict(assignment)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, c_in
        self.output_type = OPTIONS["head"][assignment["head"]].output
        self.normalization = build_normalization(assignment["normalization"], c_in)
        self.decomposition = build_decomposition(assignment["decomposition"], kernel_size)
        individual = assignment["channel"] == "individual"
        temporal = assignment["temporal"]

        def make() -> _TemporalBase:
            if temporal == "linear":
                return LinearHistory(seq_len)
            if temporal == "channel_wise_linear":
                return ChannelWiseLinearTemporal(seq_len, hidden, c_in, individual)
            if temporal == "tst_transformer":
                return TSTTemporal(seq_len, patch_len, stride, hidden, n_heads, n_layers, dropout)
            if temporal == "mamba":
                return MambaTemporal(seq_len, patch_len, stride, hidden, n_layers)
            if temporal == "gated_dilated_conv":
                return GatedConvTemporal(seq_len, hidden, n_layers)
            return MixerTemporal(seq_len, c_in, hidden, n_layers, dropout)

        self.encoders = nn.ModuleList(make() for _ in range(self.decomposition.branches))
        depth, width = self.encoders[0].out_shape
        nf = depth * width * self.decomposition.branches
        head = assignment["head"]
        if head == "flatten_forecast_head":
            self.head = FlattenHead(individual, c_in, nf, pred_len, dropout)
        elif head == "quantile_head":
            self.head = QuantileProjection(individual, c_in, nf, pred_len, dropout, quantile_levels)
        else:
            self.head = GaussianProjection(c_in, nf, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (batch, {self.seq_len}, {self.enc_in}), got {tuple(x_enc.shape)}")
        parts = self.decomposition.split(self.normalization.norm(x_enc))
        z = torch.cat([enc.encode(part) for enc, part in zip(self.encoders, parts)], dim=-1)
        return self.head.project(z, self.normalization)
