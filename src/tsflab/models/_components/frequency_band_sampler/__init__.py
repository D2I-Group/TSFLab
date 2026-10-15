"""Depth-indexed contiguous frequency-band selection over an FFT axis.

Several dual time-frequency architectures allocate different spectral bands to
different network depths so that shallow layers keep high-frequency detail and
deep layers specialize on low-frequency trend. This module provides that band
arithmetic as a paper-neutral operator, independent of any one model's encoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class HierarchicalFrequencySampler(nn.Module):
    """Assign a depth-indexed contiguous frequency band out of ``num_layers``.

    Layer ``0`` receives the highest-frequency band and layer
    ``num_layers - 1`` receives the lowest-frequency (near-DC) band. With
    ``alpha <= 1 / num_layers`` the bands tile the spectrum without overlap;
    with a larger ``alpha`` a window of relative width ``alpha`` slides from
    the high-frequency end down to the low-frequency end, overlapping between
    adjacent layers. In the tiling regime the bands partition ``[0, num_bins)``
    exactly (when ``num_bins >= num_layers``).
    """

    def __init__(self, num_layers: int, alpha: float = 1.0) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.num_layers = num_layers
        self.alpha = float(alpha)

    def band(self, num_bins: int, layer_idx: int) -> tuple[int, int]:
        """Return the half-open ``[start, end)`` bin range for ``layer_idx``."""
        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers})")
        layers = self.num_layers
        if layers == 1 or self.alpha <= 1.0 / layers:
            # Exact tiling: shared integer edges floor(n * k / L), so adjacent
            # bands meet with no gap and layer 0 ends at the last bin.
            start = (num_bins * (layers - 1 - layer_idx)) // layers
            end = (num_bins * (layers - layer_idx)) // layers
        else:
            # Sliding window of width ceil(alpha * n); the rounding keeps layer 0
            # reaching the last bin and the last layer starting at bin 0.
            start = int(num_bins * (1 - self.alpha) * (1 - layer_idx / (layers - 1)))
            end = start + math.ceil(self.alpha * num_bins - 1e-9)
        end = min(num_bins, max(end, start + 1))
        return start, end

    def sample(self, spectrum: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Slice a complex spectrum ``[batch, freq, ...]`` to its band for ``layer_idx``."""
        if spectrum.ndim < 2:
            raise ValueError("spectrum must have a leading batch axis and a frequency axis")
        start, end = self.band(spectrum.shape[1], layer_idx)
        return spectrum[:, start:end]
