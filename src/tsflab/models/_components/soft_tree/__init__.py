"""Paper-neutral differentiable binary-tree routing primitives.

The modules in this file implement only the algebra shared by several local
baselines: sigmoid split decisions, products of path probabilities, and
leaf-value interpolation.  Ensemble construction and any method-specific
training interpretation remain in each model package.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def binary_routes(depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return heap node indices and right-branch indicators for every leaf."""
    if depth < 1:
        raise ValueError("depth must be positive")
    leaves = 2**depth
    nodes = torch.empty(leaves, depth, dtype=torch.long)
    right = torch.empty(leaves, depth, dtype=torch.bool)
    for leaf in range(leaves):
        node = 0
        for level in range(depth):
            branch = bool((leaf >> (depth - level - 1)) & 1)
            nodes[leaf, level] = node
            right[leaf, level] = branch
            node = 2 * node + 1 + int(branch)
    return nodes, right


def _checked_split_mask(
    split_mask: torch.Tensor | None, split_count: int, input_dim: int
) -> torch.Tensor:
    if split_mask is None:
        split_mask = torch.ones(split_count, input_dim)
    if split_mask.shape != (split_count, input_dim):
        raise ValueError("split_mask has the wrong shape")
    if (split_mask.sum(dim=-1) == 0).any():
        raise ValueError("every split node must retain at least one feature")
    return split_mask


class SoftDecisionTree(nn.Module):
    """Interpolate leaf values with differentiable binary path probabilities.

    ``split_mask`` can enforce a fixed feature subset per node.  Supplying
    ``fixed_split_weight`` and ``fixed_threshold`` creates randomized, frozen
    split geometry while retaining learned leaf predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 3,
        temperature: float = 1.0,
        *,
        split_mask: torch.Tensor | None = None,
        fixed_split_weight: torch.Tensor | None = None,
        fixed_threshold: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or depth < 1 or temperature <= 0:
            raise ValueError("dimensions and depth must be positive; temperature must exceed zero")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.temperature = float(temperature)
        split_count = 2**depth - 1

        split_mask = _checked_split_mask(split_mask, split_count, input_dim)
        self.register_buffer("split_mask", split_mask.to(dtype=torch.float32))
        self._init_split_geometry(split_count, input_dim, fixed_split_weight, fixed_threshold)

        self.leaf_value = nn.Parameter(torch.empty(2**depth, output_dim))
        nn.init.normal_(self.leaf_value, std=0.02)
        route_nodes, route_right = binary_routes(depth)
        self.register_buffer("route_nodes", route_nodes)
        self.register_buffer("route_right", route_right)

    def _init_split_geometry(
        self,
        split_count: int,
        input_dim: int,
        fixed_split_weight: torch.Tensor | None,
        fixed_threshold: torch.Tensor | None,
    ) -> None:
        if (fixed_split_weight is None) != (fixed_threshold is None):
            raise ValueError("fixed split weights and thresholds must be supplied together")
        if fixed_split_weight is None:
            self.split_weight = nn.Parameter(torch.empty(split_count, input_dim))
            self.threshold = nn.Parameter(torch.zeros(split_count))
            nn.init.normal_(self.split_weight, std=1.0 / math.sqrt(input_dim))
        else:
            if fixed_split_weight.shape != (split_count, input_dim):
                raise ValueError("fixed_split_weight has the wrong shape")
            if fixed_threshold.shape != (split_count,):
                raise ValueError("fixed_threshold has the wrong shape")
            self.register_buffer("split_weight", fixed_split_weight.to(dtype=torch.float32))
            self.register_buffer("threshold", fixed_threshold.to(dtype=torch.float32))

    def leaf_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected [batch, {self.input_dim}], got {tuple(x.shape)}")
        weight = self.split_weight * self.split_mask
        logits = (x @ weight.transpose(0, 1) - self.threshold) / self.temperature
        right_probability = torch.sigmoid(logits)
        selected = right_probability[:, self.route_nodes]
        branch = self.route_right.unsqueeze(0)
        return torch.where(branch, selected, 1.0 - selected).prod(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaf_probabilities(x) @ self.leaf_value


class SoftObliviousTree(nn.Module):
    """Soft tree whose nodes at the same depth share one split decision."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or depth < 1 or temperature <= 0:
            raise ValueError("dimensions and depth must be positive; temperature must exceed zero")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.temperature = float(temperature)
        self.split_weight = nn.Parameter(torch.empty(depth, input_dim))
        self.threshold = nn.Parameter(torch.zeros(depth))
        self.leaf_value = nn.Parameter(torch.empty(2**depth, output_dim))
        nn.init.normal_(self.split_weight, std=1.0 / math.sqrt(input_dim))
        nn.init.normal_(self.leaf_value, std=0.02)
        bits = torch.arange(2**depth).unsqueeze(1)
        shifts = torch.arange(depth - 1, -1, -1).unsqueeze(0)
        self.register_buffer("route_right", ((bits >> shifts) & 1).bool())

    def leaf_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected [batch, {self.input_dim}], got {tuple(x.shape)}")
        right_probability = torch.sigmoid(
            (x @ self.split_weight.transpose(0, 1) - self.threshold) / self.temperature
        )
        selected = right_probability.unsqueeze(1).expand(-1, 2**self.depth, -1)
        branch = self.route_right.unsqueeze(0)
        return torch.where(branch, selected, 1.0 - selected).prod(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaf_probabilities(x) @ self.leaf_value
