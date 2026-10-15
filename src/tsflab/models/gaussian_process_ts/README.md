---
name: "GaussianProcessTS"
description: "Kernel baseline: sparse RBF-kernel GP posterior-mean approximation with learned inducing inputs and horizon targets, shared across channels. Use as a small, channel-independent reference on short training sets; not for calibrated uncertainty or complex nonlinear dynamics."
---

# GaussianProcessTS

## Idea

- Treats every channel's lookback as a query vector and compares it to `num_inducing` learned inducing inputs with an RBF kernel (`_kernel`, learned length scale).
- Solves `(K_zz + noise I)^-1` against learned inducing horizon targets to get posterior-mean coefficients, then forecasts `K_xz @ coefficients`.
- Inducing pairs, length scale and noise are fit by gradient descent; this is not exact GP regression and returns no posterior covariance.

## When to use

- As a tiny-capacity baseline (about `num_inducing * (seq_len + pred_len)` parameters) when the training window is short and larger models overfit.
- Channel-independent: one shared function for all channels, so cross-channel dependence is ignored.
- No normalization: the RBF distance acts on raw window values, so it relies on the dataset's scaling.
- Not for calibrated uncertainty (no posterior covariance) or when a probabilistic output is required.

## Configure

- `enc_in`: the dataset's channel count; only checked against the input shape, since all channels share one kernel function.

Other hyperparameters: preset defaults in `configs/models/GaussianProcessTS.toml`; tune generically.

## Differences

- Independent inducing-basis mean approximation, not exact GP regression: inducing pairs are learned by gradient descent, channels share one function, and posterior covariance or calibrated uncertainty is not returned.
- Not equivalent to any third-party GP package; no such source was inspected or copied.
