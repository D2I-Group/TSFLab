"""ALiBi-style attention bias with optional per-group periodic distance."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def periodic_alibi_bias(
    seq_len: int,
    num_heads: int,
    periods: Sequence[int | None] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an additive attention bias ``[num_heads, seq_len, seq_len]``.

    Heads are split into ``g = len(periods)`` consecutive equal groups (one group
    when ``periods`` is ``None`` or empty). For head ``k`` (1-based) of a group
    with ``n = num_heads / g`` heads the bias is ``-m_k * d(i, j)`` with slope
    ``m_k = 2 ** (-8 k / n)``. The distance ``d`` is ``|i - j|`` for a group whose
    period is ``None`` and, for a period ``P``, the triangle wave of
    ``u = |i - j| mod P``: ``u`` if ``u < P / 2`` else ``P - u``.
    """
    groups = _period_groups(seq_len, num_heads, periods)
    per_group = num_heads // len(groups)
    positions = torch.arange(seq_len, device=device)
    distance = (positions.view(1, -1) - positions.view(-1, 1)).abs().to(dtype)
    slopes = torch.pow(
        torch.tensor(2.0, dtype=dtype, device=device),
        -8.0 * torch.arange(1, per_group + 1, dtype=dtype, device=device) / per_group,
    )
    blocks = []
    for period in groups:
        shaped = _shaped_distance(distance, period)
        blocks.append(-slopes.view(-1, 1, 1) * shaped.unsqueeze(0))
    return torch.cat(blocks, dim=0)


def _period_groups(seq_len: int, num_heads: int, periods: Sequence[int | None] | None) -> list[int | None]:
    """Validate the arguments and return one period (or ``None``) per head group."""
    if seq_len < 1 or num_heads < 1:
        raise ValueError("seq_len and num_heads must be positive")
    return _checked_groups(num_heads, periods)


def _checked_groups(num_heads: int, periods: Sequence[int | None] | None) -> list[int | None]:
    groups = list(periods) if periods else [None]
    if num_heads % len(groups) != 0:
        raise ValueError("num_heads must be divisible by the number of period groups")
    if any(p is not None and p < 1 for p in groups):
        raise ValueError("periods must be positive integers or None")
    return groups


def _shaped_distance(distance: torch.Tensor, period: int | None) -> torch.Tensor:
    if period is None:
        return distance
    u = distance % period
    return torch.where(u < period / 2, u, period - u)


__all__ = ["periodic_alibi_bias"]
