"""Graph-WaveNet-style diffusion convolution over static support matrices."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodConv2d(nn.Module):
    """Apply one graph support to ``(batch, channels, nodes, time)`` data."""

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", x, support).contiguous()


class PointwiseProjection(nn.Module):
    """Project concatenated diffusion channels with a 1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DiffusionConv2d(nn.Module):
    """Concatenate zero-through-``order`` graph diffusion terms and project."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        support_len: int = 3,
        order: int = 2,
    ) -> None:
        super().__init__()
        if support_len < 1:
            raise ValueError("support_len must be positive")
        if order < 1:
            raise ValueError("order must be positive")
        self.support_len = support_len
        self.nconv = NeighborhoodConv2d()
        self.mlp = PointwiseProjection((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        """Apply supports in order and return ``(B, c_out, nodes, time)``."""
        if len(supports) != self.support_len:
            raise ValueError("support count does not match configured support_len")
        outputs = [x]
        for support in supports:
            support = support.to(device=x.device)
            term = self.nconv(x, support)
            outputs.append(term)
            for _ in range(2, self.order + 1):
                term = self.nconv(term, support)
                outputs.append(term)
        projected = self.mlp(torch.cat(outputs, dim=1))
        return F.dropout(projected, self.dropout, training=self.training)


__all__ = ["DiffusionConv2d", "NeighborhoodConv2d", "PointwiseProjection"]
