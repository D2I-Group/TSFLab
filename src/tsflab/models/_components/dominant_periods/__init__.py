"""FFT-based dominant-period discovery shared by multi-period models."""

from __future__ import annotations

import numpy as np
import torch


def dominant_periods(x: torch.Tensor, k: int = 2) -> tuple[np.ndarray, torch.Tensor]:
    """Return top-k integer periods and per-sample FFT amplitudes for BLC data."""
    if x.ndim != 3:
        raise ValueError("dominant_periods expects (batch, time, channels)")
    available = torch.fft.rfft(x, dim=1).shape[1] - 1
    if k < 1 or k > available:
        raise ValueError(f"k must be between 1 and {available}, got {k}")
    spectrum = torch.fft.rfft(x, dim=1)
    frequency_strength = abs(spectrum).mean(0).mean(-1)
    frequency_strength[0] = 0
    _, top_indices = torch.topk(frequency_strength, k)
    top_indices = top_indices.detach().cpu().numpy()
    periods = x.shape[1] // top_indices
    return periods, abs(spectrum).mean(-1)[:, top_indices]
