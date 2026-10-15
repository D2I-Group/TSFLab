---
name: "ARIMATS"
description: "Differentiable conditional ARIMA(p,1,q) recurrence with shared AR/MA coefficients fit by gradient descent. Use as a few-parameter linear baseline for trending, short or compute-limited series; not for seasonal, nonlinear, or probabilistic forecasting."
---

# ARIMATS

## Idea

- Differences the input once (d=1) and runs the ARMA recurrence over the differences, with learnable `ar_coefficients`, `ma_coefficients` and `drift`.
- Historical one-step innovations are computed from the observed differences; future innovations are zero (conditional expectation).
- Forecasts are the last observed value plus the cumulative sum of predicted differences.
- Coefficients are shared across channels and fit by gradient descent, not likelihood.

## When to use

- A sanity baseline with a handful of parameters: short training windows, tight compute, or series dominated by a random-walk-like level with linear short-term dependence.
- First differencing removes trend non-stationarity; there is no seasonal term, so strongly seasonal data needs another model.
- One coefficient set for all channels; no cross-channel modelling; no confidence intervals.

## Configure

- `enc_in`: number of channels (shape check only).
- `ar_order`, `ma_order`: generic lag orders (at least one positive); no automatic order selection.

Other hyperparameters: preset defaults in `configs/models/ARIMATS.toml`; tune generically.

## Differences

Clean-room implementation of Box and Jenkins, *Time Series Analysis: Forecasting and Control* (Holden-Day, 1970). The differencing order is fixed to one, coefficients are estimated by gradient descent and shared across channels, and unknown future innovations are their conditional zero. It does not perform likelihood fitting, order selection (ACF/PACF, AIC/BIC), stationarity transforms, seasonal ARIMA, or confidence intervals. No third-party implementation was inspected or copied.
