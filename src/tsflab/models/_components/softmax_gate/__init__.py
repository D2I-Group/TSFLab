"""Softmax feature gate: rescale features by a learned distribution over them."""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftmaxGate(nn.Module):
    """Multiply the last axis by ``softmax(Linear(x), dim=-1)``.

    The gate weights are non-negative and sum to one over the last axis, so the
    block upscales dominant features and suppresses the rest. It has no residual
    path and no normalization; callers place it after an MLP and add their own
    skip connection.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.score = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last axis {self.dim}, got {tuple(x.shape)}")
        return x * torch.softmax(self.score(x), dim=-1)


__all__ = ["SoftmaxGate"]
