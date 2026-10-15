"""Window-level spectral descriptor: spectral entropy and three band-energy ratios.

A paper-neutral summary of how a multivariate window's spectral energy is
distributed. It is parameter-free, so any gate, router, or scheduler can consume
the four statistics without owning an FFT pipeline.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SpectralDescriptor(nn.Module):
    """Map ``[batch, length, channels]`` to ``[batch, 4]`` spectral statistics.

    Per sample: remove each channel's temporal mean, take the one-sided power
    spectrum ``|rFFT|^2`` over ``K = length // 2 + 1`` bins, average it across
    channels, and smooth-normalize it to a distribution
    ``p = (P + eps) / (sum(P) + K * eps)``. The output columns are

    * normalized spectral entropy ``-sum(p log p) / log(max(K, 2))`` in ``[0, 1]``;
    * low, middle, and high band-energy ratios, the sums of ``p`` over bins
      ``[0, K//3]``, ``[K//3 + 1, 2K//3]``, and ``[2K//3 + 1, K - 1]``, which sum
      to one.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("SpectralDescriptor expects (batch, length, channels)")
        spectrum = torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1)
        power = (spectrum.real.square() + spectrum.imag.square()).mean(dim=2)
        bins = power.shape[1]
        p = (power + self.eps) / (power.sum(dim=1, keepdim=True) + bins * self.eps)
        entropy = -(p * torch.log(p + self.eps)).sum(dim=1) / math.log(max(bins, 2))
        low_end, mid_end = bins // 3 + 1, 2 * bins // 3 + 1
        return torch.stack(
            (
                entropy,
                p[:, :low_end].sum(dim=1),
                p[:, low_end:mid_end].sum(dim=1),
                p[:, mid_end:].sum(dim=1),
            ),
            dim=1,
        )
