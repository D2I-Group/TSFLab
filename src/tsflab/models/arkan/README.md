---
name: "ARKAN"
description: "Pre-fitted AR coefficients weight the last p lags; a B-spline KAN maps them to the next value; recursive rollout. Use for small univariate-style one-step or short-horizon forecasting with linear memory plus nonlinearity; not for long horizons or cross-channel structure."
---

# ARKAN

## Idea

- AR memory (Sec. III-B, Eqs. 7-8): one AR(p) coefficient per lag and channel, fitted on the training split before the network trains, applied as fixed filters (elementwise product with the last `lag` values).
- `fit_ar` uses no-intercept least squares on lag windows (official code, default) or Yule-Walker `a = R^-1 r` (Eqs. 9-12).
- `BSplineKANLayer` (Eqs. 5-6): each edge is `w_b silu(x) + w_s sum_k c_k B_k(x)` with cubic B-splines (`bspline_basis`) on a uniform grid over `[-grid_bound, grid_bound]`; the `[lag, hidden, 1]` map is shared by all channels.
- Training is one step ahead on observed lags; `forward` forecasts the horizon recursively.

## When to use

- Small, mostly univariate problems where a linear autoregressive memory explains much of the dynamics and a light nonlinear correction helps; the paper's protocol is one step ahead (`pred_len = 1`).
- Recursive rollout accumulates error for `pred_len > 1`; not designed for long horizons.
- Channels are independent (per-channel AR, shared KAN); point output only.
- The spline grid covers `[-1, 1]` by default, so inputs should be standardized.

## Configure

- `enc_in`: number of channels.
- `lag`: AR order, in `[1, seq_len]`.
- `ar_estimator`: the AR memory is fitted automatically from the training split in `training_setup`; until then the weights are ones (plain-KAN ablation).

Other hyperparameters: preset defaults in `configs/models/ARKAN.toml`; tune generically.

## Differences

Independent rewrite after reading the pinned official notebooks (MIT); `pykan` is not a dependency.

- AR memory: least squares by default (official); the paper's Yule-Walker is available.
- KAN: local B-spline layer; `pykan`'s symbolic branch, node/subnode affine scales and noise-based spline initialisation are not reproduced.
- B-splines are used although the paper's appendix lists FastKAN (RBF) for KAN baselines.
- The paper is univariate one-step; here channels are independent and `H > 1` is forecast recursively, which is not the paper's protocol.

Full detail in `reference.md`.
