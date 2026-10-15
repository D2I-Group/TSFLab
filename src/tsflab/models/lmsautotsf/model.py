"""Local LMS-AutoTSF implementation (paper Eqs. 2-13).

Independent rewrite from the paper; the pinned official repository (no
license file) was read only to resolve omissions such as the encoder layout,
the signed FFT frequency grid, and the filter initialisation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window

#: Average-pooling window between consecutive scales.
DOWN_SAMPLING_WINDOW = 2


def lagged_difference(x: torch.Tensor) -> torch.Tensor:
    """Autocorrelation feature ``x_t - x_{t-1}`` along axis 1; the first step keeps ``x_0``."""
    difference = x.clone()
    difference[:, 1:] = x[:, 1:] - x[:, :-1]
    return difference


def frequency_masks(length: int, cutoff: torch.Tensor, steepness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (5)-(6) sigmoid masks ``[length, channels]`` on the signed FFT grid.

    ``low = sigmoid(-(f - f_c) s)`` and ``high = sigmoid((f - f_c) s)`` with
    ``f`` from ``fftfreq(length)``, one learnable ``f_c`` and ``s`` per channel;
    the two masks sum to one at every frequency.
    """
    freqs = torch.fft.fftfreq(length, device=cutoff.device, dtype=cutoff.dtype).unsqueeze(-1)
    shifted = (freqs - cutoff) * steepness
    return torch.sigmoid(-shifted), torch.sigmoid(shifted)


def learnable_decomposition(
    x: torch.Tensor, cutoff: torch.Tensor, steepness: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[batch, time, channels]`` into trend (low-pass) and seasonal (high-pass) parts."""
    spectrum = torch.fft.fft(x, dim=1)
    low, high = frequency_masks(x.shape[1], cutoff, steepness)
    trend = torch.fft.ifft(spectrum * low, dim=1).real
    seasonal = torch.fft.ifft(spectrum * high, dim=1).real
    return trend, seasonal


class ScaleEncoder(nn.Module):
    """Encoder of one component at one scale (Fig. 1b, Eqs. 8-10).

    A temporal MLP over the time axis gives ``x_temp``, which is multiplied by
    the input's lagged difference; the product is added to the input, a channel
    MLP of the product is added unless channels are independent, and a linear
    map over time produces the horizon.
    """

    def __init__(
        self, length: int, pred_len: int, enc_in: int, d_model: int, dropout: float, channel_mixing: bool
    ) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(length, d_model), nn.ReLU(), nn.Linear(d_model, length), nn.Dropout(dropout)
        )
        self.channel = (
            nn.Sequential(nn.Linear(enc_in, d_model), nn.ReLU(), nn.Linear(d_model, enc_in), nn.Dropout(dropout))
            if channel_mixing
            else None
        )
        self.projection = nn.Linear(length, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[batch, length, channels]`` -> ``[batch, pred_len, channels]``."""
        x_temp = self.temporal(x.transpose(1, 2)).transpose(1, 2) * lagged_difference(x)
        mixed = x + x_temp
        if self.channel is not None:
            mixed = mixed + self.channel(x_temp)
        return self.projection(mixed.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    """LMS-AutoTSF: multi-scale learnable frequency decomposition with dual encoders."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        dropout: float = 0.1,
        num_scales: int = 4,
        channel_independence: bool = False,
        use_norm: bool = True,
        initial_cutoff: float = 0.2,
        initial_steepness: float = 10.0,
    ) -> None:
        super().__init__()
        if seq_len // DOWN_SAMPLING_WINDOW ** (num_scales - 1) < 1:
            raise ValueError("seq_len is too short for the requested number of scales")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_scales = num_scales
        self.lengths = tuple(seq_len // DOWN_SAMPLING_WINDOW**scale for scale in range(num_scales))
        self.revin = RevIN(enc_in, affine=True, enabled=use_norm)
        self.cutoff = nn.Parameter(torch.full((num_scales, enc_in), float(initial_cutoff)))
        self.steepness = nn.Parameter(torch.full((num_scales, enc_in), float(initial_steepness)))
        mixing = not channel_independence
        self.trend_encoders = nn.ModuleList(
            ScaleEncoder(length, pred_len, enc_in, d_model, dropout, mixing) for length in self.lengths
        )
        self.seasonal_encoders = nn.ModuleList(
            ScaleEncoder(length, pred_len, enc_in, d_model, dropout, mixing) for length in self.lengths
        )
        self.projection = nn.Linear(pred_len * num_scales, pred_len)

    def multi_scale(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Eq. (2): the input and its successive average-pooled (window 2) copies."""
        scales = [x]
        for _ in range(1, self.num_scales):
            pooled = F.avg_pool1d(scales[-1].transpose(1, 2), DOWN_SAMPLING_WINDOW)
            scales.append(pooled.transpose(1, 2))
        return scales

    def scale_forecasts(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Eqs. (5)-(12) per scale: decompose, encode trend and seasonal parts, and sum."""
        outputs = []
        for index, series in enumerate(self.multi_scale(x)):
            trend, seasonal = learnable_decomposition(series, self.cutoff[index], self.steepness[index])
            outputs.append(self.trend_encoders[index](trend) + self.seasonal_encoders[index](seasonal))
        return outputs

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        stacked = torch.cat(self.scale_forecasts(normalized), dim=1)
        forecast = self.projection(stacked.transpose(1, 2)).transpose(1, 2)
        return self.revin(forecast, "denorm")


__all__ = [
    "Model",
    "ScaleEncoder",
    "frequency_masks",
    "lagged_difference",
    "learnable_decomposition",
]
