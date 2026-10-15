---
name: "quantile_head"
description: "Non-crossing quantile head: a median anchor plus cumulative softplus gaps give an ascending [B, L, C, Q] quantile grid with input-dependent width. Use for tasks needing quantiles or intervals scored by pinball/CRPS; not for parametric distributions or independent per-level heads."
---

# quantile_head

## What it does

Given per-step base features `base [B, L, C, F]`, `QuantileHead(levels, F)` emits
`Q = len(levels)` quantiles that cannot cross. With `m` the index of the level
closest to 0.5:

- `anchor = Linear(F -> 1)(base)` and `gaps = softplus(Linear(F -> Q-1)(base))` (each gap >= 0).
- `out[m] = anchor`; for `i > m`: `out[i] = anchor + sum_{j=m..i-1} gap_j`; for `i < m`: `out[i] = anchor - sum_{j=i..m-1} gap_j`.

So the output is non-decreasing in the level index by construction, and interval
width depends on the input.

## When to use

Use when the task needs quantiles or prediction intervals (pinball or CRPS-style
scoring) and the quantiles must not cross. Do not use when independent per-level
heads are wanted (levels are coupled through the anchor), when the distribution
is parametric or sample-based, or when the levels must be stored with the
checkpoint (they are not).

## Interface

Public symbols: `QuantileHead`, `validate_quantile_levels`, `DEFAULT_QUANTILE_LEVELS`.

- `DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, ..., 0.9)`.
- `validate_quantile_levels(values)`: `None` returns the defaults as a list; otherwise returns the list after checking it is non-empty, each level strictly inside (0, 1), and strictly increasing; raises `ValueError` otherwise.
- `QuantileHead(quantile_levels: list[float], in_features: int = 1)`: levels are validated as above (`None` is also accepted and means the defaults, though the annotation says list; `in_features` is not range-checked). `in_features` is the trailing width of `base` (use 1 for a scalar anchor per step, after `unsqueeze(-1)`). Attributes `q`, `median_idx`, `in_features`.
- `forward(base [B, L, C, in_features]) -> [B, L, C, Q]` (any leading shape also works because only the last axis is projected, but the output is documented for rank 4). `Q == 1` returns just the anchor (`offset_proj` is then an empty `[0, F]` Linear that still appears in the state dict). Raises the `Linear` shape error when the trailing width mismatches.
- Parameters and state-dict keys: `anchor_proj.weight/bias` (`[1, F]`), `offset_proj.weight/bias` (`[Q-1, F]`). The levels are a non-persistent buffer `_levels` and are not in the state dict, so the checkpoint does not record the levels.
- Even-length level lists with 0.5 absent pick the level closest to 0.5 (ties go to the lower index, up to float32 rounding of the stored levels) as the anchor; the anchor is then not an exact median.
