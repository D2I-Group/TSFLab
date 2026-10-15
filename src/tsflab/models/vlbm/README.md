---
name: "VLBM"
description: "Variational latent-basis forecaster: prior-sampled basis states drive a base stream and a cosine-graph residual stream, trained with a posterior-prior KL. Use for multivariate or sensor data under distribution shift with calendar marks; not for deterministic outputs (forecasts are sampled) or known-topology graph use."
---

# VLBM

## Idea

- `VariableEmbedding` (Eq. 17): each variable becomes one vector from its normalized window, time-of-day and day-of-week tables at the window's last step, and a node identity table.
- `GraphStructuredEncoder` (Eqs. 20-23): additive graph attention (identity mask, as in the official code) then mean and log-variance of `M` basis activations; the prior sees `(X, X)`, the posterior `(X, Y)` with `Y` the embedded future.
- Latent basis (Eqs. 18-19, 24-25): `w = mu + sigma * eps` is sampled from the prior in training and inference, and `z = w B` drives the generator, so forecasts are stochastic.
- `BaseResidualGenerator`: a base stream `[Z, X]` with pointwise residual blocks (Eq. 29) and a residual stream `[X - Z, X]` that mixes variables over `A = Softmax(TopK(0.8 cos(z_i, z_j) + 0.2 I))` (Eq. 31); one linear head maps their sum to the horizon under non-affine `revin`.
- Objective (Eq. 33, `training_objective`): forecast loss on the prior path plus `kl_weight * KL(q || p)`; the posterior never enters the forecast.

## When to use

- Designed for out-of-distribution robustness: shifts between training and test distributions.
- Variables interact through a learned latent cosine graph, so correlated channels or sensors benefit.
- Calendar slots come from the marks, so time-of-day and weekday signal can be used.
- Forecasts are sampled even at evaluation; not for tasks that need deterministic point outputs or calibrated quantiles.

## Configure

- `enc_in`: must equal the channel (or node) count; must match `num_nodes` on graph datasets. `adj_mx` is ignored.
- `interval`: sampling interval in minutes; the time-of-day table has `1440 / interval` slots (10 for 10-minute Weather, 60 for hourly).
- `top_k`: neighbours kept per row of the latent graph; must be `<= enc_in`.

Train with the `mae` loss, as in the official scripts. Other hyperparameters: preset defaults in `configs/models/VLBM.toml`; tune generically.

## Differences

- Follows the code, not the paper, where they differ: no ridge projection (base input `[Z, X]`, residual input `[X - Z, X]`), one shared basis, identity attention mask, and one head on the summed streams.
- Calendar slots come from the timestamps of the window's last step, not row positions in the file.
- The configured criterion is unmasked (the official loss swaps arguments and ends up plain MAE).
- The normalization standard deviation is detached (catalog `revin`), which leaves parameter gradients unchanged.
- Section V-D ablation variants are not implemented.
