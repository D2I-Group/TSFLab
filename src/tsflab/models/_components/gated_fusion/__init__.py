"""Learnable sigmoid gate that convexly blends two equal-shaped embeddings.

Several forecasters combine a "global" and a "local" (or a "temporal" and a
"variate") representation with a data-dependent convex gate,
``gate = sigmoid(Wa(a) + Wb(b)); out = gate * a + (1 - gate) * b``. This
component isolates that fusion so paper-specific branches feeding it stay
model-local.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Blend two same-shaped embeddings with a learned per-position gate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.gate_a = nn.Linear(dim, dim)
        self.gate_b = nn.Linear(dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError("a and b must share the same shape")
        gate = torch.sigmoid(self.gate_a(a) + self.gate_b(b))
        return gate * a + (1.0 - gate) * b
