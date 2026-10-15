---
name: "MQRNN"
description: "LSTM encoder with a global MLP emitting horizon contexts and a shared local MLP producing non-crossing quantiles from calendar covariates. Use for multi-horizon quantile forecasting where calendar features matter; not for point-only tasks, cross-channel interaction, or static covariates."
---

# MQRNN

## Idea

- A shared `nn.LSTM` encodes each series together with historical temporal covariates (from `x_mark_enc`).
- `global_decoder` jointly maps the final state and every known-future covariate (from `x_mark_dec`) to horizon-specific and horizon-agnostic contexts (`decode_contexts`).
- One `local_decoder` MLP, shared over horizons, combines those contexts with the matching future covariate.
- `quantile_head` produces a monotone, non-crossing quantile grid `(B, pred_len, C, Q)`; `output_type = "quantile"`, trained with the pinball (`quantile`) loss. `quantile_levels` default to `evaluation.quantile_levels`.

## When to use

- Multi-horizon forecasts that need quantiles (prediction intervals) rather than a point value.
- Series driven by calendar effects, where time features of the history and of the forecast horizon carry signal.
- Not when only a point forecast is wanted, when cross-channel interaction matters (channels share one encoder independently), or when static item covariates or other exogenous inputs are needed (only time marks are used).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `future_covariate_size`: must equal the width of `x_mark_enc` / `x_mark_dec`, i.e. the time-feature width for the dataset's sampling frequency; `0` disables covariates. Future marks must cover `pred_len`.

Other hyperparameters: preset defaults in `configs/models/MQRNN.toml`; tune generically. Pair with `[training] loss = "quantile"`.

## Differences

- Local implementation of Section 3.2 of Wen, Torkkola, Narayanaswamy, Madeka (2017, arXiv 1711.11053): `(c_1..c_K, c_a) = m_G(h_t, x_future)`, then `q_hat_k = m_L(c_k, c_a, x_future_k)`; no author code exists or was inspected.
- Uses TSFLab's monotone `QuantileHead`; the paper does not impose this parameterization.
- Static item covariates and the forking-sequences training objective are not supported by the standard forecaster call.
- With `features = "MS"` only the last channel's quantiles are returned.
