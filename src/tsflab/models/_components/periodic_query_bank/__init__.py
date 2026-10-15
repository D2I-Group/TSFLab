"""A learnable table of per-phase vectors gathered into phase-aligned windows.

Several long-term forecasting papers (CycleNet's residual cycle, TQNet's
temporal query) keep one learnable vector per position of a fixed period and
read out a contiguous, phase-aligned window of that table for every sample in
a batch. This component isolates that gather-only indexing so new models can
reuse it without re-deriving the modular-arithmetic offsets. Paper-specific
uses of the retrieved window (removed as a residual, used as an attention
query, and so on) stay model-local.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PeriodicQueryBank(nn.Module):
    """One learnable vector per phase of a fixed period, indexed by offset.

    Holds a zero-initialized table of shape ``(period, channels)``. Given a
    per-sample starting phase and a requested window length, returns the
    table rows visited by advancing that phase, wrapping modulo ``period``.
    """

    def __init__(self, period: int, channels: int) -> None:
        super().__init__()
        if period < 1 or channels < 1:
            raise ValueError("period and channels must be positive")
        self.period = period
        self.channels = channels
        self.table = nn.Parameter(torch.zeros(period, channels))

    def forward(self, start_phase: torch.Tensor, length: int) -> torch.Tensor:
        """Gather a phase-aligned window for every sample in the batch.

        Args:
            start_phase: integer tensor of shape ``(batch,)`` giving the
                table row each sample begins reading from.
            length: number of consecutive (wrapped) rows to read.

        Returns:
            Tensor of shape ``(batch, length, channels)``.
        """
        if start_phase.ndim != 1:
            raise ValueError("start_phase must be a 1-D tensor of per-sample phases")
        if length < 1:
            raise ValueError("length must be positive")
        offsets = torch.arange(length, device=start_phase.device)
        indices = (start_phase[:, None] + offsets[None, :]).remainder(self.period)
        return self.table[indices]
