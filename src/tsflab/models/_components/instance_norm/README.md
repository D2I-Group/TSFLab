---
name: "instance_norm"
description: "Parameter-free instance standardization: subtract each window's mean and divide by its std over time, then invert on the forecast. Use for non-stationary series without a learnable affine; not when RevIN's affine or last-value centering is needed (use revin)."
---

# instance_norm

## What it does

`standardize(x, eps=1e-5, detach_mean=True, detach_std=False)` returns `(z, mean, std)` with,
over the time axis (dim 1):

- `mean = x.mean(1, keepdim=True)`, detached when `detach_mean=True`;
- `std = sqrt(var(x - mean, dim=1, unbiased=False) + eps)`, detached when `detach_std=True`;
- `z = (x - mean) / std`.

`destandardize(y, mean, std)` returns `y * std + mean`; the statistics broadcast
over the forecast's time axis, so `y` may be longer or shorter than `x`.

The operation order is the one of the Time-Series-Library forecasters, so a
model that switches to this unit keeps bit-identical outputs and gradients.

## When to use

Use for non-stationary windows whose level or scale drifts, when the model
should see standardized inputs and forecast in the original scale, with no
parameters or state. Use `revin` instead when the paper adds a learnable affine
or centers on the last value, and a patch-wise normalizer when statistics must
be per patch.

## Interface

- `standardize(x: Tensor, eps: float = 1e-5, detach_mean: bool = True, detach_std: bool = False) -> tuple[Tensor, Tensor, Tensor]`
- `destandardize(y: Tensor, mean: Tensor, std: Tensor) -> Tensor`

No parameters, buffers, or state; functions are pure and device-agnostic.
