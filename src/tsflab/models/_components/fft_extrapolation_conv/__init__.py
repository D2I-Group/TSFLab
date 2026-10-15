"""History-to-horizon complex frequency-domain convolution with zero padding."""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.weight_set_router import mix_weight_sets


class FFTExtrapolationConv(nn.Module):
    """Map ``(batch, channels, input_length)`` to a horizon by a long convolution.

    The history is zero padded to ``input_length + output_length - 1`` (plus a
    guard band of about one percent on each side), transformed with ``rfft``,
    multiplied bin by bin with a complex weight, shifted by a complex bias, and
    transformed back; the last ``output_length`` samples before the right guard
    band are returned. Equivalent to a time-domain convolution whose kernel
    spans the whole history and horizon. Each of ``num_sets`` weight sets owns
    one complex weight and one complex bias per frequency bin; an optional
    ``(num_sets, channels)`` mixing matrix forms each channel's weight as a
    linear combination of the sets.
    """

    def __init__(self, input_length: int, output_length: int, num_sets: int = 1) -> None:
        super().__init__()
        if min(input_length, output_length, num_sets) < 1:
            raise ValueError("lengths and num_sets must be positive")
        self.input_length = input_length
        self.output_length = output_length
        self.num_sets = num_sets
        self.guard = max(math.floor((input_length + output_length - 1) / 100), 1)
        self.padded_length = input_length + output_length - 1 + 2 * self.guard
        self.num_bins = self.padded_length // 2 + 1
        # Initialized to the all-pass-at-DC "average" filter: weight[:, 0] = 1.
        real_weight = torch.zeros(num_sets, self.num_bins)
        real_weight[:, 0] = 1.0
        self.real_weight = nn.Parameter(real_weight)
        self.imag_weight = nn.Parameter(torch.zeros(num_sets, self.num_bins))
        self.real_bias = nn.Parameter(torch.zeros(num_sets, self.num_bins))
        self.imag_bias = nn.Parameter(torch.zeros(num_sets, self.num_bins))

    def forward(self, x: torch.Tensor, mixing: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_length:
            raise ValueError("FFTExtrapolationConv expects (batch, channels, input_length)")
        weight = torch.complex(self.real_weight, self.imag_weight)
        bias = torch.complex(self.real_bias, self.imag_bias)
        if mixing is None:
            if self.num_sets != 1:
                raise ValueError("mixing is required when num_sets > 1")
            weight, bias = weight[0], bias[0]  # (F,) shared by every channel
        else:
            if mixing.shape != (self.num_sets, x.shape[1]):
                raise ValueError("mixing must have shape (num_sets, channels)")
            mixing = mixing.to(weight.dtype)
            weight = mix_weight_sets(weight, mixing)
            bias = mix_weight_sets(bias, mixing)
        padded = nn.functional.pad(x, (self.guard, self.output_length - 1 + self.guard))
        spectrum = torch.fft.rfft(padded) * weight + bias
        series = torch.fft.irfft(spectrum, n=self.padded_length)
        return series[..., -self.output_length - self.guard : -self.guard]


__all__ = ["FFTExtrapolationConv"]
