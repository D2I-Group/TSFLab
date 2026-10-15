"""Shared monotone, non-crossing quantile output head.

Converts a per-(B, pred_len, C) base representation into a rank-4 quantile
tensor (B, pred_len, C, Q) whose last axis is sorted ascending BY CONSTRUCTION
and ordered to match the configured quantile levels. Quantiles cannot cross
because lower/upper quantiles are built from the median anchor by SUBTRACTING /
ADDING strictly non-negative, cumulative, input-conditioned offsets.

Contract:
- output[..., m] == anchor at the median index m (the level closest to 0.5).
- output[..., i] is monotone non-decreasing in i (ascending levels).
- interval width is input-conditioned: every offset is softplus(Linear(base)),
  a learned function of the per-step base features (NOT a constant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def validate_quantile_levels(
    values: list[float] | tuple[float, ...] | None,
) -> list[float]:
    """Return a validated, strictly increasing quantile-level list."""
    levels = list(values) if values is not None else list(DEFAULT_QUANTILE_LEVELS)
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantile levels must be non-empty and lie strictly inside (0, 1)")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantile levels must be strictly increasing")
    return levels


class QuantileHead(nn.Module):
    """Monotone quantile head producing a non-crossing (B, L, C, Q) grid.

    Parameters
    ----------
    quantile_levels : list[float]
        Quantile levels, ascending, matching ``config.evaluation.quantile_levels``
        (e.g. the 9 deciles ``[0.1, ..., 0.9]``). ``Q = len(quantile_levels)``.
        The output's last axis is ordered to match this list exactly.
    in_features : int
        Width of the per-step base feature vector passed to ``forward`` along
        the trailing dimension. For an anchor-only input (a scalar per
        ``(B, L, C)`` step) use ``in_features = 1``; for a richer per-step
        feature vector pass that vector's width.
    """

    def __init__(self, quantile_levels: list[float], in_features: int = 1) -> None:
        super().__init__()
        levels = validate_quantile_levels(quantile_levels)
        # Ensure ascending order is the contract; the head emits in this order.
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32), persistent=False)
        self.q = len(levels)
        # Median index: the level closest to 0.5. The anchor passes through here.
        self.median_idx = int(torch.argmin((self._levels - 0.5).abs()).item())
        self.in_features = in_features
        # Anchor projection: base features -> the median quantile value.
        self.anchor_proj = nn.Linear(in_features, 1)
        # Offset projections: one strictly-non-negative offset per quantile
        # GAP. There are (Q - 1) gaps; below-median gaps are subtracted via a
        # downward cumulative sum, above-median gaps are added via an upward
        # cumulative sum. softplus makes each gap >= 0 => non-crossing.
        self.offset_proj = nn.Linear(in_features, max(self.q - 1, 0))

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        """Build the non-crossing quantile grid.

        Parameters
        ----------
        base : torch.Tensor
            Per-step base features of shape ``(B, L, C, in_features)``. When the
            caller has only a scalar anchor per step ``(B, L, C)``, it must
            ``unsqueeze(-1)`` to ``(B, L, C, 1)`` before calling.

        Returns
        -------
        torch.Tensor
            Quantile grid ``(B, L, C, Q)``, ascending along the last axis,
            equal to the anchor at ``median_idx``.
        """
        anchor = self.anchor_proj(base).squeeze(-1)  # (B, L, C)
        if self.q == 1:
            return anchor.unsqueeze(-1)  # (B, L, C, 1)
        gaps = F.softplus(self.offset_proj(base))  # (B, L, C, Q-1) >= 0

        m = self.median_idx
        out = torch.empty((*anchor.shape, self.q), dtype=anchor.dtype, device=anchor.device)
        out[..., m] = anchor
        # Above median: q_{i} = anchor + sum_{j=m..i-1} gap_j  (gap index = j)
        cum_up = 0.0
        for i in range(m + 1, self.q):
            cum_up = cum_up + gaps[..., i - 1]
            out[..., i] = anchor + cum_up
        # Below median: q_{i} = anchor - sum_{j=i..m-1} gap_j
        cum_dn = 0.0
        for i in range(m - 1, -1, -1):
            cum_dn = cum_dn + gaps[..., i]
            out[..., i] = anchor - cum_dn
        return out  # (B, L, C, Q)
