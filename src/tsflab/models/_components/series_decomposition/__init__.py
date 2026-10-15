"""Edge-padded moving-average decomposition for ``(batch, time, channel)`` data."""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgePaddedMovingAverage(nn.Module):
    """Smooth a series after repeating its first and last observations."""

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if stride < 1:
            raise ValueError("stride must be positive")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the temporal moving average with edge-value padding."""
        padding = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding, 1)
        end = x[:, -1:, :].repeat(1, padding, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """Split ``(B, L, C)`` values into residual and moving-average trend."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = EdgePaddedMovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(residual, trend)`` without changing the input layout."""
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


__all__ = [
    "EdgePaddedMovingAverage",
    "SeriesDecomposition",
]
