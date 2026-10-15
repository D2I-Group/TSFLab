"""Gated top-k expert routing shared by DUET-style and DynamicTMoE-style mixtures.

Two paper-neutral pieces are extracted here:

- ``GatingMLP`` reproduces a two-layer ``Linear -> GELU -> Linear`` gate with an
  optional trainable-noise term added to the logits before softmax during
  training. This is the exact gate DUET's ``DistributionalRouter`` used.
- ``topk_dense_mix`` reproduces the "zero outside top-k, blend in a floor of
  the dense weights, renormalize" sparsification shared by DUET's post-router
  mixing step and DynamicTMoE's ``routing_weights`` concentration step. Both
  models compute dense softmax weights, zero every non-selected expert, add a
  small floor fraction of the dense weights back in (to keep every expert's
  gradient alive), and renormalize so the weights sum to one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingMLP(nn.Module):
    """Two-layer gate: ``Linear -> GELU -> Linear`` with optional trainable noise.

    Reproduces DUET's ``DistributionalRouter`` gate: a dense softmax over
    ``experts`` logits, with an optional per-expert trainable noise scale
    (``softplus``-transformed) added before softmax while training.
    """

    def __init__(self, in_features: int, experts: int, hidden: int, noisy: bool = True) -> None:
        super().__init__()
        self.noisy = noisy
        self.network = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Linear(hidden, experts))
        self.noise_scale = nn.Parameter(torch.zeros(experts)) if noisy else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return dense softmax weights over ``experts`` for ``[*, in_features]`` input."""
        logits = self.network(features)
        if self.training and self.noise_scale is not None:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise_scale)
        return torch.softmax(logits, -1)


def topk_dense_mix(weights: torch.Tensor, k: int, floor: float) -> torch.Tensor:
    """Zero non-top-k experts, blend back a ``floor`` of the dense weights, renormalize.

    ``weights`` is a dense per-expert distribution (typically a softmax output)
    over its last axis. The top-``k`` entries are kept at their dense value,
    every other entry is zeroed, ``floor * weights`` is added back everywhere
    so every expert keeps a live gradient, and the result is renormalized to
    sum to one over the last axis. Raises ``ValueError`` unless
    ``1 <= k <= num_experts`` and ``floor >= 0``.
    """
    if not 1 <= k <= weights.shape[-1]:
        raise ValueError(f"k must be in [1, {weights.shape[-1]}] (number of experts), got {k}")
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}")
    top_indices = weights.topk(k, dim=-1).indices
    sparse = torch.zeros_like(weights).scatter(-1, top_indices, weights.gather(-1, top_indices))
    combined = sparse + floor * weights
    return combined / combined.sum(-1, keepdim=True)


__all__ = ["GatingMLP", "topk_dense_mix"]
