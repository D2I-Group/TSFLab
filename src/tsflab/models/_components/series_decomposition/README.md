---
name: "series_decomposition"
description: "Edge-padded moving average over time and the split x = (x - MA(x)) + MA(x) for [B, L, C] series (Autoformer, DLinear). Use for series with a smooth trend or drifting level that should be modelled apart from the seasonal part; not for even windows, learned trends, or other time axes."
---

# series_decomposition

## What it does

`EdgePaddedMovingAverage(kernel_size, stride)` smooths each channel over time
with a window of `k = kernel_size` after repeating the first and last time step
`(k - 1) / 2` times at the two ends, so with stride 1 the output keeps the
length. `SeriesDecomposition(kernel_size)` uses it with stride 1:

`trend = MA(x)`, `residual = x - trend`, returned as `(residual, trend)`, so
`residual + trend == x`. Callers often call the residual "seasonal".

## When to use

Use for series whose smooth trend or slowly drifting level should be modelled
separately from the faster seasonal/residual part (Autoformer- and DLinear-style
trend/residual splitting of `[B, L, C]` tensors with an odd window). Do not use
when the paper specifies an even window or different boundary handling, when the
time axis is not axis 1, or when the trend must be learned.

## Interface

`EdgePaddedMovingAverage(kernel_size, stride=1)`

- `kernel_size` (odd int >= 1): window; even or < 1 raises `ValueError`
  ("kernel_size must be a positive odd integer"). `stride` (int >= 1): raises
  `ValueError` otherwise.
- `forward(x)`: `x` `[batch, length, channels]` (rank 3 required by the
  indexing). Output length is `length` when `stride == 1`, and
  `floor((length + k - 1 - k) / stride) + 1` otherwise, so with `stride > 1` it
  no longer matches the input. `kernel_size > length` works (edge repeats).
- Attributes `kernel_size`, `avg` (`nn.AvgPool1d`).

`SeriesDecomposition(kernel_size)`

- `forward(x)` returns `(residual, trend)` as two tensors of the shape of `x`.
  Attribute `moving_avg`.
- Both modules have no parameters or buffers (empty state dict), are
  differentiable, and keep dtype/device. They are stateless.
