---
name: "AutoRegressiveTS"
description: "One shared linear map from the whole lag window to all horizons, applied to every channel independently. Use as the minimal linear baseline for short training data or tight budgets; not for nonlinear, non-stationary, or cross-channel dynamics."
---

# AutoRegressiveTS

## Idea

- A single `nn.Linear(seq_len, pred_len)` maps lagged observations directly to every forecast step; no recursive rollout.
- Coefficients are shared across channels, so it is not a vector autoregression.
- Minimal classical autoregressive baseline: no normalization, decomposition or nonlinearity.

## When to use

- Sanity baseline and floor for any benchmark; very few parameters, so it suits short training windows and tight compute.
- No normalization: level shifts between training and test hurt it directly.
- No cross-channel terms and no nonlinearity; point output only.

## Configure

- `enc_in`: number of channels (shape check only).
- No other data-dependent parameter; the lag order is `seq_len`.

Other hyperparameters: none beyond the preset in `configs/models/AutoRegressiveTS.toml`.

## Differences

Independent implementation of the classical AR description (Box and Jenkins, *Time Series Analysis: Forecasting and Control*, Holden-Day, 1970); no external implementation inspected or copied. Unlike a recursively fitted one-step AR model (OLS or Yule-Walker), it learns one direct map from the complete lag window to all horizons by gradient descent. Coefficients are shared across channels and do not form a vector autoregression.
