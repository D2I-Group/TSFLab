"""Kernel-free scalar-state selective scan and a 2-D grid state mixer.

Some Mamba-style forecasters use a *scalar* per-channel state (one decay per
channel rather than a bank of ``d_state`` states) and then spatially mix that
state across a 2-D grid of positions -- for example a (time-patch, variate)
grid -- before it is read out. This module isolates those two paper-neutral
pieces:

* :func:`diagonal_selective_scan` runs the sequential recurrence
  ``h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t`` without any
  CUDA-only kernel, mirroring the portability contract of
  ``tsflab.models._components.mamba`` but for the degenerate ``d_state == 1`` case
  and without folding the ``C`` read-out or ``D`` skip into the scan, so a
  caller can post-process the raw state before reading it out.
* :class:`GridStateMixer` propagates that state across a 2-D grid with a
  depthwise convolution. It is a portable stand-in for offset-based
  deformable convolutions (for example DCNv4), which require a custom CUDA
  kernel; using a fixed local receptive field keeps the same "mix a state
  with its spatial neighbors" semantics on CPU at the cost of the learned
  sampling offsets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def diagonal_selective_scan(u: torch.Tensor, delta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate a scalar-state selective scan over the trailing length axis.

    Args:
        u: input tensor, shape ``(batch, channels, length)``.
        delta: input-dependent step size, shape ``(batch, channels, length)``.
        a: per-channel log-decay rate (already negative), shape ``(channels,)``.
        b: input-dependent gate, shape ``(batch, channels, length)``.

    Returns:
        Hidden-state trajectory ``h``, shape ``(batch, channels, length)``,
        with ``h[..., t]`` the state after reading position ``t``.
    """
    if u.shape != delta.shape or u.shape != b.shape:
        raise ValueError("u, delta, and b must share one (batch, channels, length) shape")
    if a.ndim != 1 or a.shape[0] != u.shape[1]:
        raise ValueError("a must have shape (channels,) matching u's channel axis")
    if u.shape[-1] == 0:
        raise ValueError("length must be at least 1; the scan of an empty sequence is undefined")
    decay = torch.exp(delta * a.view(1, -1, 1))
    drive = delta * b * u
    state = torch.zeros_like(u[..., 0])
    outputs = []
    for t in range(u.shape[-1]):
        state = decay[..., t] * state + drive[..., t]
        outputs.append(state)
    return torch.stack(outputs, dim=-1)


class GridStateMixer(nn.Module):
    """Depthwise 2-D convolution mixing a channel-first state grid locally.

    Consumes and returns ``(batch, channels, height, width)``; ``height`` and
    ``width`` are caller-defined grid axes (for example time-patch and
    variate position).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("channels must be positive and kernel_size a positive odd integer")
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4:
            raise ValueError("grid must have shape (batch, channels, height, width)")
        return self.conv(grid)
