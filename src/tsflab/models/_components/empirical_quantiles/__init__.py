"""Linear-interpolated empirical quantiles of a stack of forecast samples."""

from __future__ import annotations

import torch


def empirical_quantiles(samples: torch.Tensor, levels: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Linear-interpolated quantiles over the sample axis ``dim``, moved to a trailing axis.

    ``samples`` holds ``K`` draws along ``dim`` (for example ``[K, B, L, C]`` with
    ``dim=0`` or ``[B, K, L, C]`` with ``dim=1``); ``levels`` is a 1-D tensor of
    ``Q`` levels in ``[0, 1]``. Sorting the draws as ``x_(0) <= ... <= x_(K-1)``,
    level ``q`` maps to ``h = q (K - 1)`` and returns
    ``x_(floor h) + (h - floor h) (x_(ceil h) - x_(floor h))`` (Hyndman-Fan
    definition 7, ``torch.quantile(..., interpolation="linear")``) without the
    input-size limit of ``torch.quantile``. The result keeps the remaining axes
    in order and appends the level axis: ``[B, L, C, Q]`` for both examples.
    """
    ordered = samples.movedim(dim, 0).sort(dim=0).values
    position = levels.to(samples.dtype) * (ordered.shape[0] - 1)
    lower = position.floor().long()
    upper = position.ceil().long()
    weight = (position - lower.to(samples.dtype)).view((-1,) + (1,) * (ordered.dim() - 1))
    low, high = ordered[lower], ordered[upper]
    return (low + (high - low) * weight).movedim(0, -1)


__all__ = ["empirical_quantiles"]
