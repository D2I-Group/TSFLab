"""Low-rank weight sharing: softmax routing of a few weight sets onto channels."""

from __future__ import annotations

import torch
from torch import nn


class WeightSetRouter(nn.Module):
    """Learn a ``(num_sets, channels)`` routing matrix normalized over the sets.

    ``forward(temperature)`` returns ``softmax(R / temperature, dim=0)``; each
    column is the convex combination of the ``num_sets`` weight sets assigned
    to one channel. The temperature is supplied by the caller (it is typically
    annealed during training).
    """

    def __init__(self, num_sets: int, channels: int) -> None:
        super().__init__()
        if num_sets < 1 or channels < 1:
            raise ValueError("num_sets and channels must be positive")
        self.num_sets = num_sets
        self.channels = channels
        self.logits = nn.Parameter(torch.randn(num_sets, channels))

    def forward(self, temperature: float = 1.0) -> torch.Tensor:
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        return torch.softmax(self.logits / temperature, dim=0)


def mix_weight_sets(weights: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
    """Combine ``(num_sets, *shape)`` weights into ``(channels, *shape)``.

    ``out[c] = sum_s routing[s, c] * weights[s]`` for a ``(num_sets, channels)``
    routing matrix.
    """
    if routing.ndim != 2 or routing.shape[0] != weights.shape[0]:
        raise ValueError("routing must be (num_sets, channels) matching weights")
    return torch.einsum("sc,s...->c...", routing, weights)


__all__ = ["WeightSetRouter", "mix_weight_sets"]
