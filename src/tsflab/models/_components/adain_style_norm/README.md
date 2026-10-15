---
name: "adain_style_norm"
description: "Parameter-free AdaIN: standardize [B, L, C] over L with detached mean/std, then rescale to a supplied style mean and std. Use for re-injecting per-sample or learned statistics into features of non-stationary series; not for reversible self-normalization (use revin) or length-1 inputs."
---

# adain_style_norm

## What it does

`AdaptiveInstanceNorm1d(eps)` standardizes a feature tensor along its sequence
axis and then re-injects caller-supplied statistics instead of learned affine
parameters. For `x` of shape `[B, L, C]`:

- `mu = mean_L(x)` and `sigma = std_L(x, unbiased=True) + eps`, both detached;
- `y = (x - mu) / sigma * style_std + style_mean`.

The module owns no parameters; the "style" is an input. Gradients flow through
`x` (as the numerator and through the constant `1/sigma` scale) and through
`style_mean` and `style_std`, but not through `mu` and `sigma`.

## When to use

Use when features should be standardized per sample over the sequence axis and
re-scaled to statistics produced elsewhere (for example per-sample series
moments or a learned style branch). Do not use when the statistics should be
the input's own and later undone (use `revin`), when the normalized axis is not
axis 1, when the standardization statistics should receive gradients, or for
inputs of length 1.

## Interface

`AdaptiveInstanceNorm1d(eps=1e-5)`

- `eps` (float): added to the standard deviation (not under the square root).
  No range validation is done; use a positive value.
- No parameters, buffers, or state-dict keys (`state_dict()` is empty).

`forward(x, style_mean, style_std)`

- `x`: floating tensor `[batch, length, channels]`. Raises `ValueError` if
  `x.ndim != 3`.
- `style_mean`, `style_std`: broadcastable to `[batch, 1, channels]` (a scalar
  tensor also broadcasts). They are not validated and are not detached.
  `style_std` is used as given (no eps, no positivity check).
- Returns a tensor with the broadcast shape of the result (the shape of `x`
  for the intended `[batch|1, 1, channels]` statistics) on the device and dtype
  of the inputs.
- Uses the unbiased standard deviation, so `length == 1` yields NaN (and a
  PyTorch warning). Normalization is always over axis 1.
