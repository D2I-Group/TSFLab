---
name: "empirical_quantiles"
description: "Sort K forecast samples along a sample axis and read Q quantile levels by linear interpolation (Hyndman-Fan type 7), level axis last. Use for sample-based probabilistic forecasters (diffusion, VAE) scored on a quantile grid; not for direct quantile heads, closed-form distribution heads, or samples with NaNs."
---

# empirical_quantiles

## What it does

Turns a stack of `K` forecast draws into a quantile forecast. With the draws
sorted along the sample axis as `x_(0) <= ... <= x_(K-1)` and each level `q`
mapped to the fractional rank `h = q (K - 1)`:

`Q_q = x_(floor h) + (h - floor h) * (x_(ceil h) - x_(floor h))`.

This is `torch.quantile(samples, levels, dim, interpolation="linear")` with the
level axis moved last, computed by an explicit sort and gather so it has no
input-size limit. It is the output stage of sample-based probabilistic
forecasters (diffusion samplers, latent push-forward models, VAEs) scored with
quantile losses.

## When to use

Use when a model produces Monte Carlo forecast draws and must emit a quantile
grid ordered like `config.evaluation.quantile_levels`. Do not use when the
model predicts quantiles directly (use `quantile_head`) or predicts
distribution parameters with a closed-form quantile function (see
`gaussian_parameter_head`), or when samples contain NaNs (they sort last and
corrupt the upper quantiles).

## Interface

`empirical_quantiles(samples, levels, dim=0) -> Tensor`

- `samples`: floating tensor with `K >= 1` draws along `dim` (any rank >= 1).
  The four diffusion consumers pass `[K, B, L, C]` with `dim=0`; `ppm` and
  `voldyvae` pass `[B, K, L, C]` with `dim=1`.
- `levels`: 1-D tensor of `Q` levels in `[0, 1]`; it is cast to
  `samples.dtype` before use. Levels are not validated: values outside `[0, 1]`
  produce out-of-range indices (an indexing error or wrap-around for negative
  ranks). Unsorted levels are allowed and keep their order.
- Returns `[*other axes in their order, Q]`; for both layouts above the output
  is `[B, L, C, Q]`. When the input levels are ascending the output is
  non-decreasing along the last axis.
- Stateless: no parameters, buffers or state-dict keys; differentiable in
  `samples` (gradient flows through the sort to the two neighbouring order
  statistics of each level), not in `levels`.
