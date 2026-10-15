"""Parameter-free instance normalization over the time axis (the non-stationary normalization of TSLib)."""

from __future__ import annotations

import torch


def standardize(
    x: torch.Tensor, eps: float = 1e-5, detach_mean: bool = True, detach_std: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x - mean) / std`` with the window mean and std over dim 1, plus both statistics.

    The mean is ``x.mean(1, keepdim=True)``; the std is ``sqrt(var(x - mean, dim=1,
    unbiased=False) + eps)``. ``detach_mean`` and ``detach_std`` stop their gradients.
    """
    mean = x.mean(1, keepdim=True)
    if detach_mean:
        mean = mean.detach()
    centered = x - mean
    std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + eps)
    if detach_std:
        std = std.detach()
    return centered / std, mean, std


def destandardize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert :func:`standardize` on a forecast: ``y * std + mean``, broadcast over its time axis."""
    return y * std + mean
