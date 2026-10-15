"""Lossless single-level Haar discrete wavelet transform along the last axis.

The (orthonormal, Haar/db1) wavelet is the simplest case where a one-level
decomposition of an even-length signal needs no boundary padding: adjacent
sample pairs are rotated into one low-pass ("approximation") and one
high-pass ("detail") coefficient, and the inverse transform is the exact
algebraic inverse of that rotation. This makes the pair below a paper-neutral
"map a series into wavelet sub-series and back" building block for
wavelet-decomposition forecasters.

Odd-length inputs are handled by replicate-padding one extra sample before the
forward transform and dropping the corresponding trailing sample after the
inverse transform; this is documented, not hidden, behavior.
"""

from __future__ import annotations

import math

import torch
from torch import nn

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class HaarDWT1D(nn.Module):
    """Forward single-level Haar DWT along the last axis."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(approx, detail)`` each of length ``ceil(T / 2)``."""
        if x.shape[-1] < 2:
            raise ValueError("last axis must have length >= 2")
        padded = x if x.shape[-1] % 2 == 0 else torch.cat([x, x[..., -1:]], dim=-1)
        even = padded[..., 0::2]
        odd = padded[..., 1::2]
        approx = (even + odd) * _INV_SQRT2
        detail = (even - odd) * _INV_SQRT2
        return approx, detail


class HaarIDWT1D(nn.Module):
    """Inverse single-level Haar DWT along the last axis."""

    def forward(self, approx: torch.Tensor, detail: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """Reconstruct the time-domain signal from ``(approx, detail)``.

        ``length`` truncates the reconstruction (used to undo the forward
        transform's replicate-padding of an odd-length input).
        """
        if approx.shape != detail.shape:
            raise ValueError("approx and detail must share the same shape")
        even = (approx + detail) * _INV_SQRT2
        odd = (approx - detail) * _INV_SQRT2
        reconstructed = torch.stack([even, odd], dim=-1).reshape(*approx.shape[:-1], approx.shape[-1] * 2)
        if length is not None:
            reconstructed = reconstructed[..., :length]
        return reconstructed


__all__ = ["HaarDWT1D", "HaarIDWT1D"]
