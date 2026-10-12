"""Local WFTNet: Wavelet-Fourier Transform blocks balanced by a periodicity coefficient.

Paper map (Liu et al., ICASSP 2024, arXiv 2309.11319): encoder normalization,
value/position/calendar embedding and temporal alignment to ``T_e = T_s + T_p``
(Section 3.2); WFTBlock (Section 3.3) with a Fourier branch (Eq. 1, top-k periods
folded into 2D maps) and a continuous-wavelet branch (Eq. 2, Morlet CWT over a set
of scales) sharing one Time-Frequency Inception Block; amplitude-softmax fusion of
the Fourier maps, a strip-shaped convolution over the wavelet scales, and the
Periodicity-Weighted Coefficient fusion ``X^{l+1} = a^n X_f + (1 - a^n) X_w``
(Section 3.4) plus the residual input; linear projection and de-normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.dominant_periods import dominant_periods
from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.inception_block import InceptionBlock2d
from tsflab.models._components.marks import adapt_tslib_marks, tslib_time_feature_dimension
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window

# Real Morlet mother wavelet psi(t) = exp(-t^2 / 2) cos(5 t), tabulated on
# [-8, 8] with 2**10 points and integrated by rectangles (the pywt "morl" grid the
# official ptwt dependency samples).
MORLET_BOUND = 8.0
MORLET_POINTS = 2**10


def periodicity_weighted_coefficient(series: torch.Tensor, first_bins: int = 5) -> float:
    """Section 3.4: ``alpha = mean_c max_{i<=m} a_i^2 / sum_{i<=m} a_i^2``.

    ``series`` is ``[time, channels]``; ``a_i`` are the FFT amplitudes of each
    channel and ``m = first_bins`` counts the lowest bins (the DC bin included, as
    in the official script). The value is a dataset statistic computed once and
    passed to the model as ``period_coeff``.
    """
    if series.ndim != 2 or first_bins < 1:
        raise ValueError("series must be [time, channels] and first_bins positive")
    energy = torch.fft.fft(series.double(), dim=0).abs().pow(2)[:first_bins]
    return float((energy.max(dim=0).values / energy.sum(dim=0)).mean())


def wavelet_scales(wavelet_scale: int, count: int = 8) -> torch.Tensor:
    """CWT scales ``2 ** linspace(-1, wavelet_scale, count)``."""
    return torch.pow(2.0, torch.linspace(-1.0, float(wavelet_scale), count, dtype=torch.float64))


def morlet_cwt_operator(length: int, scales: torch.Tensor) -> torch.Tensor:
    """Linear CWT operator ``W`` with ``coef[s] = x @ W[s]``, shape ``[S, length, length]``.

    Eq. (2) discretised as in pywt-style CWT: the integrated mother wavelet is
    resampled at each scale, linearly convolved with the signal, differentiated,
    scaled by ``-sqrt(s)`` and centre-cropped to the signal length. The transform
    is linear and fixed for a given length, so it is tabulated once.
    """
    grid = torch.linspace(-MORLET_BOUND, MORLET_BOUND, MORLET_POINTS, dtype=torch.float64)
    step = float(grid[1] - grid[0])
    integrated = torch.cumsum(torch.exp(-grid.pow(2) / 2) * torch.cos(5 * grid), 0) * step
    impulses = torch.eye(length, dtype=torch.float64)
    operators = []
    for scale in scales.tolist():
        span = scale * 2 * MORLET_BOUND
        index = torch.floor(torch.arange(span + 1, dtype=torch.float64) / (scale * step)).long()
        kernel = integrated[index[index < MORLET_POINTS]].flip(0)
        size = length + kernel.numel() - 1
        full = torch.fft.irfft(
            torch.fft.rfft(impulses, n=size, dim=-1) * torch.fft.rfft(kernel, n=size), n=size, dim=-1
        )
        coef = -math.sqrt(scale) * torch.diff(full, dim=-1)
        excess = (coef.shape[-1] - length) / 2
        if excess > 0:
            coef = coef[..., math.floor(excess): coef.shape[-1] - math.ceil(excess)]
        operators.append(coef)
    return torch.stack(operators)


class WFTBlock(nn.Module):
    """One Wavelet-Fourier Transform block on ``[B, T_e, D]`` (Section 3.3)."""

    def __init__(
        self, total_length: int, d_model: int, d_ff: int, num_kernels: int, top_k: int,
        wavelet_scale: int, period_coeff: float, pwc_power: int, scale_count: int = 8,
    ) -> None:
        super().__init__()
        self.total_length, self.top_k = total_length, top_k
        self.pwc_weight = float(period_coeff) ** pwc_power
        self.inception = nn.Sequential(
            InceptionBlock2d(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock2d(d_ff, d_model, num_kernels),
        )
        # Strip-shaped depthwise convolution compressing the scale axis.
        self.scale_conv = nn.Conv2d(d_model, d_model, kernel_size=(scale_count, 1), groups=d_model)
        self.register_buffer(
            "cwt_operator",
            morlet_cwt_operator(total_length, wavelet_scales(wavelet_scale, scale_count)).float(),
            persistent=False,
        )

    def fourier_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Top-k period folding, shared inception, amplitude-softmax sum: ``X_f``."""
        batch, length, channels = x.shape
        periods, amplitudes = dominant_periods(x, self.top_k)
        maps = []
        for period in (int(p) for p in periods):
            rows = math.ceil(length / period)
            padded = F.pad(x, (0, 0, 0, rows * period - length))
            image = padded.reshape(batch, rows, period, channels).permute(0, 3, 1, 2)
            folded = self.inception(image).permute(0, 2, 3, 1).reshape(batch, -1, channels)
            maps.append(folded[:, :length])
        weights = torch.softmax(amplitudes, dim=1)
        return (torch.stack(maps, dim=-1) * weights[:, None, None, :]).sum(-1)

    def wavelet_map(self, x: torch.Tensor) -> torch.Tensor:
        """Morlet CWT of every embedding channel: ``[B, T_e, D]`` -> ``[B, D, S, T_e]``."""
        return torch.einsum("bdt,stu->bdsu", x.transpose(1, 2), self.cwt_operator.to(x.dtype))

    def wavelet_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Shared inception on the time-scale map, then the strip conv: ``X_w``."""
        return self.scale_conv(self.inception(self.wavelet_map(x))).squeeze(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.total_length:
            raise ValueError(f"WFTBlock expects {self.total_length} steps, got {x.shape[1]}")
        fused = self.pwc_weight * self.fourier_branch(x) + (1 - self.pwc_weight) * self.wavelet_branch(x)
        return fused + x


class Model(nn.Module):
    """WFTNet forecaster: embedding, temporal alignment, stacked WFTBlocks, projection."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period_coeff: float,
        d_model: int = 512,
        d_ff: int = 2048,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        wavelet_scale: int = 1,
        pwc_power: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if top_k > (seq_len + pred_len) // 2:
            raise ValueError("top_k exceeds the available non-DC frequencies of seq_len + pred_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        total = seq_len + pred_len
        self.revin = RevIN(enc_in, affine=False)
        # Value (circular conv) + sinusoidal position + hourly timeF calendar features.
        self.embedding = DataEmbedding(
            enc_in, d_model, embed_type="timeF", freq="h", dropout=dropout,
            time_feature_dim=tslib_time_feature_dimension("h"),
        )
        self.predict_linear = nn.Linear(seq_len, total)
        self.blocks = nn.ModuleList(
            WFTBlock(total, d_model, d_ff, num_kernels, top_k, wavelet_scale, period_coeff, pwc_power)
            for _ in range(e_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, enc_in)

    def encode(self, normalized: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """Embedding then the temporal alignment ``T_s -> T_e`` over the time axis."""
        marks = adapt_tslib_marks(x_mark_enc, embed_type="timeF", freq="h")
        if marks is not None:
            marks = marks.to(normalized.dtype)
        embedded = self.embedding(normalized, marks)
        return self.predict_linear(embedded.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        hidden = self.encode(self.revin(x_enc, "norm"), x_mark_enc)
        for block in self.blocks:
            # One LayerNorm shared by every block, as in the official encoder loop.
            hidden = self.layer_norm(block(hidden))
        output = self.revin(self.projection(hidden), "denorm")
        return output[:, -self.pred_len:]
