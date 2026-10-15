"""WaveNet-style causal dilated gated activation unit.

This module extracts the paper-neutral primitive shared by dilated
convolutional spatiotemporal forecasters: left (causal) padding sized to the
convolution's dilation and kernel width, followed by
``tanh(filter(x)) * sigmoid(gate(x))``. Graph mixing, residual and skip
connections, and any post-gate normalization stay model-local because they
vary per paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_pad(x: torch.Tensor, dilation: int, kernel_size: int) -> torch.Tensor:
    """Left-pad the trailing (temporal) axis for a causal dilated convolution.

    Supports rank-3 ``(batch, channels, time)`` and rank-4
    ``(batch, channels, nodes, time)`` tensors; only the last axis is padded.
    """
    padding = dilation * (kernel_size - 1)
    if x.dim() == 3:
        return F.pad(x, (padding, 0))
    if x.dim() == 4:
        return F.pad(x, (padding, 0, 0, 0))
    raise ValueError("causal_pad expects a rank-3 or rank-4 tensor")


def gated_dilated_conv(x: torch.Tensor, filter_conv: nn.Module, gate_conv: nn.Module) -> torch.Tensor:
    """Apply the WaveNet gated activation unit to ``x``.

    ``filter_conv`` and ``gate_conv`` are caller-owned ``Conv1d`` or ``Conv2d``
    modules (so parameter names and shapes stay on the calling layer); their
    ``kernel_size`` and ``dilation`` (last axis) set the causal left-padding.
    Returns ``tanh(filter_conv(padded)) * sigmoid(gate_conv(padded))``.
    """
    kernel_size = filter_conv.kernel_size[-1]
    dilation = filter_conv.dilation[-1]
    padded = causal_pad(x, dilation, kernel_size)
    return torch.tanh(filter_conv(padded)) * torch.sigmoid(gate_conv(padded))


__all__ = ["causal_pad", "gated_dilated_conv"]
