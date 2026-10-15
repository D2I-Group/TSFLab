---
name: "GaussianMLP"
description: "In-repository flatten-MLP baseline predicting per-step Gaussian location and positive scale, trained by negative log-likelihood. Use as a simple probabilistic baseline when the task needs a predictive distribution; not for point-forecast leaderboards or many-channel, long windows."
---

# GaussianMLP

## Idea

- Flattens the `seq_len * enc_in` window and passes it through a ReLU/dropout MLP (`backbone`): `h_0 = vec(X)`, `h_l = Dropout(ReLU(W_l h_{l-1} + b_l))`.
- `gaussian_parameter_head` emits `loc = W_mu h` and `scale = softplus(W_sigma h) + eps` per horizon step and channel, stacked as `(B, pred_len, C, 2)`.
- Declares `output_type = "distribution"`, `distribution_family = "gaussian"`; trained with `nll_gaussian` (pair with `[training] loss = "nll_gaussian"`).
- A standard parametric baseline (Gaussian likelihood head as popularized by DeepAR-style models); no single canonical paper. See `deepar` for an RNN-based distribution forecaster.

## When to use

- When the task needs a predictive distribution (Gaussian NLL, CRPS) and a cheap reference point for richer probabilistic models.
- Mixes channels only through the flattened input; the first layer has `seq_len * enc_in` inputs, so it grows with window length and channel count.
- Not for heavy-tailed or multimodal targets (single Gaussian per step) or when cross-channel / cross-horizon covariance matters (not modeled).
- No instance normalization: relies on the dataset's scaling.

## Configure

- `enc_in`: the dataset's channel count (input width `seq_len * enc_in`; with `features = "MS"` one output channel).

Other hyperparameters: preset defaults in `configs/models/GaussianMLP.toml`; tune generically.

## Differences

- Intentional in-repository baseline, not an external paper reproduction.
- Predicts independent Gaussian location/scale pairs; cross-channel and cross-horizon covariance are not modeled.
