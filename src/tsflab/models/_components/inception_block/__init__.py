"""TimesNet-style 2D inception block: the mean of same-padded odd square convolutions."""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionBlock2d(nn.Module):
    """Mean of ``num_kernels`` same-padded square 2D convolutions with kernel sizes
    ``1, 3, ..., 2 num_kernels - 1``.

    With ``init_weight=True`` each kernel gets Kaiming-normal fan-out (ReLU gain)
    weights and zero bias after construction; with ``False`` the PyTorch
    ``nn.Conv2d`` default initialization is kept.
    """

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int, init_weight: bool = True) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i) for i in range(num_kernels)
        )
        if init_weight:
            for conv in self.kernels:
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([conv(x) for conv in self.kernels], dim=-1).mean(-1)


__all__ = ["InceptionBlock2d"]
