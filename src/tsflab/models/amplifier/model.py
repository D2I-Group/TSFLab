"""Clean-room Amplifier implementation from the published equations."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import SeriesDecomposition


def flipped_spectrum(values: torch.Tensor) -> torch.Tensor:
    """Return the paper's frequency-reversed one-sided spectrum (Eq. 5)."""
    return torch.flip(torch.fft.rfft(values, dim=1), dims=(1,))


class ComplexFrequencyProjection(nn.Module):
    """Complex length mapping used by the restoration block (Eq. 8)."""

    def __init__(self, input_bins: int, output_bins: int) -> None:
        super().__init__()
        self.real = nn.Linear(input_bins, output_bins)
        self.imag = nn.Linear(input_bins, output_bins)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        real = self.real(spectrum.real.transpose(1, 2)).transpose(1, 2)
        imag = self.imag(spectrum.imag.transpose(1, 2)).transpose(1, 2)
        return torch.complex(real, imag)


class SemiChannelInteraction(nn.Module):
    """Commonality/specificity temporal refinement from Eqs. 10--11."""

    def __init__(self, length: int, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.commonality = nn.Sequential(
            nn.Linear(channels, channels), nn.LeakyReLU(), nn.Linear(channels, 1)
        )
        self.common_temporal = nn.Sequential(
            nn.Linear(length, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, length)
        )
        self.specific_temporal = nn.Sequential(
            nn.Linear(length, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, length)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        common = self.commonality(values)
        common = self.common_temporal(common.transpose(1, 2)).transpose(1, 2)
        specific = values - common
        specific = self.specific_temporal(specific.transpose(1, 2)).transpose(1, 2)
        return common + specific


class Model(nn.Module):
    """Forecast-only Amplifier with Eqs. 5--13 represented explicitly."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int = 128,
        sci: bool = True,
        moving_average: int = 25,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.revin = RevIN(enc_in)
        self.sci = SemiChannelInteraction(seq_len, enc_in, hidden_size) if sci else None
        self.decomposition = SeriesDecomposition(moving_average)
        self.seasonal_forecaster = nn.Sequential(
            nn.Linear(seq_len, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, pred_len)
        )
        self.trend_forecaster = nn.Sequential(
            nn.Linear(seq_len, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, pred_len)
        )
        self.restoration = ComplexFrequencyProjection(seq_len // 2 + 1, pred_len // 2 + 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(
                f"x_enc must have shape (batch, {self.seq_len}, {self.channels})"
            )
        normalized = self.revin(x_enc, "norm")
        mirrored = flipped_spectrum(normalized)
        amplified = torch.fft.irfft(
            torch.fft.rfft(normalized, dim=1) + mirrored,
            n=self.seq_len,
            dim=1,
        )
        if self.sci is not None:
            amplified = self.sci(amplified)
        seasonal, trend = self.decomposition(amplified)
        forecast = self.seasonal_forecaster(seasonal.transpose(1, 2)).transpose(1, 2)
        forecast = forecast + self.trend_forecaster(trend.transpose(1, 2)).transpose(1, 2)
        restored = torch.fft.rfft(forecast, dim=1) - self.restoration(mirrored)
        output = torch.fft.irfft(restored, n=self.pred_len, dim=1)
        return self.revin(output, "denorm")
