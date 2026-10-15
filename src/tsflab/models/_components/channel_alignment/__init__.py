"""Deterministic trailing-channel alignment for model input adapters."""

from __future__ import annotations

import torch


def fit_channels(values: torch.Tensor, width: int) -> torch.Tensor:
    """Slice or right-pad the final axis to exactly ``width`` channels."""
    if width < 1:
        raise ValueError("width must be positive")
    if values.shape[-1] >= width:
        return values[..., :width]
    padding = values.new_zeros((*values.shape[:-1], width - values.shape[-1]))
    return torch.cat((values, padding), dim=-1)
