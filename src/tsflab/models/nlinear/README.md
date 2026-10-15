---
name: "NLinear"
description: "LTSF-Linear baseline that subtracts the last observed value, applies one linear map from lookback to horizon, and adds the value back. Use as a cheap baseline for long-horizon forecasting with level shifts between train and test; not for nonlinear dynamics, cross-channel interaction, or probabilistic output."
---

# NLinear

## Idea

- `center_on_last_value` removes the last observation and `restore_last_value` adds it back, which counters distribution shift.
- `channel_wise_linear` is a single `seq_len` to `pred_len` linear map, shared across channels unless `individual=True`.
- No decomposition or nonlinearity; it is the minimal LTSF-Linear variant with level normalization.

## When to use

- Series whose level drifts or shifts between training and test windows; last-value centering lets the linear map work on the relative shape.
- A fast, low-capacity baseline for long-horizon forecasting, or when training data or compute are limited.
- Not when the dynamics are strongly nonlinear, when cross-channel interaction or exogenous/calendar inputs matter (channels are independent, marks are ignored), or when quantiles are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (it sizes the per-channel maps when `individual=True` and is checked against the input).

Other hyperparameters: preset defaults in `configs/models/NLinear.toml`; tune generically.

## Differences

- Paper-driven local implementation; the LTSF-Linear repository is reference-only and no source file was copied or adapted.
- The last observation is detached and subtracted from every history step; the level is restored to every horizon step.
- The optional per-channel map (`individual=True`) is a local extension.
