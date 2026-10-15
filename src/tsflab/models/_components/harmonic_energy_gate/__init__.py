"""Per-channel harmonic-to-total spectral energy ratio for dual-branch gating.

A dual time/frequency-domain model can weigh its two branches by how periodic
each input channel currently is: a channel whose spectral energy concentrates
on a fundamental frequency and its low-order harmonics is weighted toward a
frequency-domain branch, while a channel with diffuse, non-harmonic energy is
weighted toward a time-domain branch. This module computes that ratio as a
paper-neutral gate, independent of any one model's branches or fusion rule.
"""

from __future__ import annotations

import torch
from torch import nn


class HarmonicEnergyGate(nn.Module):
    """Return each channel's harmonic-energy share of its total spectral energy.

    Given ``x`` shaped ``[batch, length, channels]``, the gate removes the
    per-sample mean, takes the real FFT, and locates each channel's dominant
    non-DC frequency bin within a low-frequency guard band. The energy of that
    fundamental and its next ``num_harmonics - 1`` harmonics, relative to the
    channel's total spectral energy, is returned as a weight in ``[0, 1]``.
    """

    def __init__(self, num_harmonics: int = 3, low_freq_guard: int = 3) -> None:
        super().__init__()
        if num_harmonics < 1:
            raise ValueError("num_harmonics must be positive")
        if low_freq_guard < 1:
            raise ValueError("low_freq_guard must be positive")
        self.num_harmonics = num_harmonics
        self.low_freq_guard = low_freq_guard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a ``[batch, 1, channels]`` harmonic-energy ratio in ``[0, 1]``."""
        if x.ndim != 3:
            raise ValueError("HarmonicEnergyGate expects (batch, length, channels)")
        num_bins = x.shape[1] // 2 + 1
        if num_bins <= self.low_freq_guard:
            raise ValueError("sequence too short for the configured low_freq_guard")
        spectrum = torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1)
        amplitude = spectrum.abs()

        candidate = amplitude.clone()
        candidate[:, : self.low_freq_guard, :] = 0.0
        tail_start = max(self.low_freq_guard, num_bins // self.num_harmonics)
        candidate[:, tail_start:, :] = 0.0
        _, fundamental = torch.max(candidate, dim=1, keepdim=True)

        harmonic_energy = torch.zeros_like(fundamental, dtype=amplitude.dtype)
        for order in range(1, self.num_harmonics + 1):
            bins = torch.clamp(order * fundamental, max=num_bins - 1)
            harmonic_energy = harmonic_energy + torch.gather(amplitude, 1, bins) ** 2

        total_energy = (amplitude**2).sum(dim=1, keepdim=True) + 1e-5
        return harmonic_energy / total_energy
