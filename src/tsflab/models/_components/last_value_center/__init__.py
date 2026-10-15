"""NLinear-style last-observed-value centering for ``(B, L, C)`` histories."""

from __future__ import annotations

import torch


def center_on_last_value(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the detached final timestep from every step of ``x``.

    Returns ``(centered, level)`` where ``level`` is ``x[:, -1:, :]`` detached
    from the autograd graph, so gradients cannot flow through the additive
    level term back into whatever produced ``x``. Call this before a
    forecasting head and pair it with :func:`restore_last_value` on the
    head's output.
    """
    level = x[:, -1:, :].detach()
    return x - level, level


def restore_last_value(head_output: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """Add the centering ``level`` back onto a head's forecast."""
    return head_output + level


__all__ = ["center_on_last_value", "restore_last_value"]
