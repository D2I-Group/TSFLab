---
name: "revin"
description: "Reversible instance normalization: per-window mean/std (or last-value) normalization with optional learnable affine, inverted on the forecast. Use for series whose level or scale drifts across windows or between train and test; not for heads that change channel count or statistics that need gradients."
---

# revin

## What it does

`RevIN(num_features)` removes the per-instance level and scale of a history
window before a model, then puts them back on the model output. For one instance
`x` with time axis `t` and per-channel statistics taken over all axes between
batch and channels:

- `norm`: `mu = mean_t(x)` (or `x[:, -1]` when `subtract_last=True`),
  `sigma = sqrt(var_t(x, unbiased=False) + eps)`, `y = (x - mu) / sigma`, then
  `y = y * gamma + beta` when `affine=True`.
- `denorm`: undo the affine, `z = (y - beta) / (gamma + eps^2)`, then
  `z * sigma + mu`, using the `mu`/`sigma` cached by the latest `norm` call.

The statistics are detached, so gradients do not flow through the normalizer
statistics. The module is a stateful pair: `denorm` is only valid after `norm`.

## When to use

Use when level or scale drifts across windows or between training and
evaluation periods (non-stationary series, train/validation level or scale
shift): each window is standardized and the forecast is mapped back. Do not use
when the output channel count differs from the input (statistics are
per-channel), when denormalization must happen on a different instance than the
one that normalized, when statistics should carry gradients, or when a paper
specifies non-detached or patch-wise statistics.

## Interface

`RevIN(num_features, eps=1e-5, affine=True, subtract_last=False, enabled=True)`

- `num_features` (int >= 1): channel width; must equal `values.shape[-1]`.
- `eps` (float > 0): variance floor added inside the square root; it also
  appears as `eps^2` in the denominator of the affine inversion.
- `affine` (bool): adds learnable `affine_weight` (init 1) and `affine_bias`
  (init 0), each `[num_features]`. These are the only parameters and the only
  state-dict keys; `affine=False` registers none.
- `subtract_last` (bool): center on the last time step instead of the mean;
  the scale is still the standard deviation.
- `enabled` (bool): when false, `forward` returns its input unchanged (same
  object) for both modes and builds no statistics.

`forward(values, mode)` with `mode` in `{"norm", "denorm"}`:

- `values`: floating tensor `[batch, time, channels]` or higher rank with the
  channel axis last (`ndim >= 3`). Statistics reduce over every axis from 1 to
  `ndim - 2`. In `denorm` the time length may differ from the history (for
  example the forecast horizon) because the cached statistics broadcast over it.
- Returns a tensor of the same shape and dtype/device as `values`.
- Raises `ValueError` for `ndim < 3`, a channel-width mismatch, or an unknown
  mode; `RuntimeError` when `denorm` precedes any `norm`. All of these checks are
  skipped when `enabled=False`. Constructor: `ValueError` for `num_features < 1`
  or `eps <= 0`.
- Rank > 3: statistics and the `subtract_last` center reduce over axes
  `1..ndim-2` (see reference.md, Higher-rank inputs).
- Cached `_center` and `_scale` are plain attributes (not buffers): they are not
  in the state dict, follow whichever instance last called `norm`, and make an
  instance unsafe to share across concurrent forward passes.
