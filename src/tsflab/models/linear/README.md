---
name: "Linear"
description: "Single learnable history-to-horizon linear map applied along time to each channel (LTSF-Linear). Use as the minimal baseline for long-term forecasting and efficiency checks; not for data with strong level shifts (no normalization) or nonlinear dynamics."
---

# Linear

## Idea

- One affine map `W` of shape `pred_len x seq_len` (`ChannelWiseLinear`) forecasts every channel independently.
- Weights are shared across channels by default; `individual=True` gives each channel its own map (a repository extension).
- No normalization, decomposition or nonlinearity; the plain baseline behind DLinear and NLinear.

## When to use

- The reference baseline for long-term forecasting: the paper shows it beats many Transformer forecasters on nine datasets.
- Very cheap; suits tight compute budgets and sanity checks.
- No normalization: level shifts between train and test hurt it (use NLinear or RevIN-wrapped models instead).
- Cannot model cross-channel interactions or nonlinear dynamics.

## Configure

- `enc_in`: must equal the channel count (one map per channel when `individual = true`).

Other hyperparameters: preset defaults in `configs/models/Linear.toml`; tune generically.

## Differences

- Paper-driven local implementation of `X_hat_i = W X_i` as one temporal affine map shared across channels; the external repository is reference-only.
- `individual` is a documented repository extension with one map per channel.
- Paper preprocessing and reported benchmark values are outside code-validation scope.
