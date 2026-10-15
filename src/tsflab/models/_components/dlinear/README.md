---
name: "dlinear"
description: "DLinear backbone: split [B, L, C] by an edge-padded moving average into trend and remainder, project each with its own channel-wise linear map, and sum to [B, H, C]. Use for a cheap channel-independent baseline or base forecaster; not for covariates or time marks, cross-channel interaction, or variable-length inputs."
---

# dlinear

## What it does

`DLinearBackbone(c_in, seq_len, pred_len, kernel_size, individual)` forecasts
by decomposing the history, projecting each part linearly over time, and
adding:

`r, t = SeriesDecomposition(kernel_size)(x)` (`t` is the edge-padded moving
average, `r = x - t`), then
`y = ChannelWiseLinear_seasonal(r^T) + ChannelWiseLinear_trend(t^T)`, transposed
back to `[B, H, C]`.

## When to use

Use as a strong, cheap channel-independent baseline or as a base forecaster
inside a larger model. Do not use when covariates or time marks must enter, when
cross-channel interaction is required, or for variable-length inputs.

## Interface

`DLinearBackbone(c_in, seq_len, pred_len, kernel_size=25, individual=False)`

- `c_in` (int >= 1): channels; `seq_len` (int >= 1): history length;
  `pred_len` (int >= 1): horizon.
- `kernel_size` (odd int >= 1, default 25): moving-average window; even or
  non-positive values raise `ValueError` from `EdgePaddedMovingAverage`. Larger
  than `seq_len` is accepted (edge padding repeats the end values).
- `individual` (bool): per-channel (`True`) or shared (`False`) projections.
- Attributes: `decomposition`, `seasonal_projection`, `trend_projection`,
  `seq_len`, `pred_len`, `channels`, `individual`.
- State-dict keys: `seasonal_projection.linear.{weight,bias}` and
  `trend_projection.linear.{weight,bias}` (shared), or
  `seasonal_projection.linears.{c}.*` / `trend_projection.linears.{c}.*`
  (individual). The decomposition has no parameters.
- `forward(x)`: `x` `[batch, seq_len, c_in]` -> `[batch, pred_len, c_in]`. The
  backbone does not validate `x`'s shape itself; `ChannelWiseLinear` raises
  `ValueError` if channels or length mismatch.
- No special weight initialization: the projections use the default `nn.Linear`
  initialization (no `1/seq_len` constant init is applied).
