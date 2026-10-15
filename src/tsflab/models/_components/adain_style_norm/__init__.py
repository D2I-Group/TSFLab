"""Adaptive instance normalization: rescale features to external statistics.

Non-stationarity-aware forecasters normalize a feature tensor to zero mean
and unit standard deviation over its sequence axis, then re-inject externally
supplied ("style") mean and standard-deviation statistics instead of a fixed
affine transform. This is the same computation as image-style AdaIN, applied
to a ``(batch, length, channels)`` embedding: the statistics act as a
non-stationary, sample-adaptive substitute for LayerNorm's learned affine
parameters. Deriving those style statistics from raw series moments is
paper-specific and stays model-local.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveInstanceNorm1d(nn.Module):
    """Normalize over the sequence axis, then rescale to given statistics."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN.

        Args:
            x: features, shape ``(batch, length, channels)``.
            style_mean: target mean, broadcastable to ``(batch, 1, channels)``.
            style_std: target standard deviation, broadcastable likewise.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, length, channels)")
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True).detach() + self.eps
        normalized = (x - mean) / std
        return normalized * style_std + style_mean
