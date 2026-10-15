"""Gaussian location/scale projection with explicit positivity semantics."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianParameterHead(nn.Module):
    """Project features to independent Gaussian location and positive scale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        eps: float = 1e-6,
        scale_transform: Literal["softplus", "log1pexp"] = "softplus",
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        if scale_transform not in {"softplus", "log1pexp"}:
            raise ValueError(f"unsupported scale transform: {scale_transform}")
        self.eps = eps
        self.scale_transform = scale_transform
        self.loc_layer = nn.Linear(in_features, out_features)
        self.scale_layer = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc_layer(x)
        raw_scale = self.scale_layer(x)
        if self.scale_transform == "softplus":
            scale = F.softplus(raw_scale)
        else:
            # Preserve the official DeepAR expression for reference comparison.
            scale = torch.log(1 + torch.exp(raw_scale))
        return loc, scale + self.eps
