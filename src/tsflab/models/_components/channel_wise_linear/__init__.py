"""Linear projection over the last axis, shared or independent by channel."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelWiseLinear(nn.Module):
    """Map ``(batch, channels, input_length)`` to a forecast length.

    The shared path applies one affine map to every channel. The individual
    path owns one affine map per channel. Normalization, decomposition, and
    layout conversion remain caller responsibilities.
    """

    def __init__(
        self,
        input_length: int,
        output_length: int,
        channels: int,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.channels = channels
        self.individual = individual
        if individual:
            self.linears = nn.ModuleList(nn.Linear(input_length, output_length) for _ in range(channels))
        else:
            self.linear = nn.Linear(input_length, output_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("ChannelWiseLinear expects (batch, channels, input_length)")
        if x.shape[1:] != (self.channels, self.input_length):
            raise ValueError("input shape does not match the configured channels/length")
        if self.individual:
            return torch.stack(
                [linear(x[:, index, :]) for index, linear in enumerate(self.linears)],
                dim=1,
            )
        return self.linear(x)
