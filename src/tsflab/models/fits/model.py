"""Paper-driven local implementation of frequency interpolation forecasting."""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.instance_norm import destandardize, standardize


class ComplexFrequencyInterpolation(nn.Module):
    """Learn a complex affine interpolation between retained frequency bands."""

    def __init__(self, input_bins: int, output_bins: int, channels: int, individual: bool):
        super().__init__()
        shape = (channels if individual else 1, output_bins, input_bins)
        scale = 1.0 / math.sqrt(input_bins)
        self.real_weight = nn.Parameter(torch.empty(shape).uniform_(-scale, scale))
        self.imag_weight = nn.Parameter(torch.empty(shape).uniform_(-scale, scale))
        self.real_bias = nn.Parameter(torch.zeros(shape[0], output_bins))
        self.imag_bias = nn.Parameter(torch.zeros(shape[0], output_bins))
        self.channels = channels
        self.individual = individual

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        weight_real = self.real_weight if self.individual else self.real_weight.expand(self.channels, -1, -1)
        weight_imag = self.imag_weight if self.individual else self.imag_weight.expand(self.channels, -1, -1)
        bias_real = self.real_bias if self.individual else self.real_bias.expand(self.channels, -1)
        bias_imag = self.imag_bias if self.individual else self.imag_bias.expand(self.channels, -1)
        real = torch.einsum("bci,coi->bco", spectrum.real, weight_real)
        real = real - torch.einsum("bci,coi->bco", spectrum.imag, weight_imag) + bias_real
        imag = torch.einsum("bci,coi->bco", spectrum.real, weight_imag)
        imag = imag + torch.einsum("bci,coi->bco", spectrum.imag, weight_real) + bias_imag
        return torch.complex(real, imag)


class Model(nn.Module):
    """Normalize, low-pass filter, interpolate frequencies, and invert the FFT."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int,
                 individual: bool = False, cut_freq: int = 24) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.total_len = seq_len + pred_len
        self.input_bins = min(cut_freq, seq_len // 2 + 1)
        ratio = self.total_len / seq_len
        self.output_bins = min(max(1, int(round(self.input_bins * ratio))), self.total_len // 2 + 1)
        self.interpolation = ComplexFrequencyInterpolation(
            self.input_bins, self.output_bins, enc_in, individual
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        normalized, mean, scale = standardize(x_enc, 1e-5, detach_mean=True, detach_std=True)
        spectrum = torch.fft.rfft(normalized.transpose(1, 2), dim=-1)[..., : self.input_bins]
        interpolated = self.interpolation(spectrum)
        padded = torch.zeros(
            x_enc.shape[0], self.enc_in, self.total_len // 2 + 1,
            dtype=interpolated.dtype, device=x_enc.device,
        )
        padded[..., : self.output_bins] = interpolated
        reconstructed = torch.fft.irfft(padded, n=self.total_len, dim=-1)
        reconstructed = reconstructed * (self.total_len / self.seq_len)
        forecast = reconstructed[..., -self.pred_len :].transpose(1, 2)
        return destandardize(forecast, mean, scale)
