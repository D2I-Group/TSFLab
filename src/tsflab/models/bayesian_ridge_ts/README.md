---
name: "BayesianRidgeTS"
description: "Linear lag regression whose weights carry a learned Gaussian prior precision, trained as a differentiable MAP objective. Use as a regularized linear baseline for short training data or tight budgets; not for predictive uncertainty or nonlinear dynamics."
---

# BayesianRidgeTS

## Idea

- A shared `nn.Linear(seq_len, pred_len)` forecasts every channel from its own lags.
- `log_weight_precision` is learned; the forward pass sets `aux_loss` to the Gaussian weight-prior negative log-density (`0.5 precision ||w||^2 - 0.5 n log precision`), turning training into MAP estimation.
- No evidence maximization, no observation-precision inference, no predictive uncertainty.

## When to use

- A ridge-regularized linear floor: the learned prior shrinks weights, which helps when training windows are few relative to `seq_len * pred_len` weights.
- Despite the Bayesian name and tag, it returns point forecasts only.
- Channel-independent with no normalization; level shifts hurt it directly.

## Configure

- `enc_in`: number of channels (shape check only).
- No other data-dependent parameter.

Other hyperparameters: preset defaults in `configs/models/BayesianRidgeTS.toml`; tune generically.

## Differences

Clean-room implementation using the Gaussian weight prior of MacKay, "Bayesian Interpolation", *Neural Computation* 4(3):415-447, 1992 (doi:10.1162/neco.1992.4.3.415). It performs gradient-trained MAP forecasting with a learned positive prior precision and a shared channel-wise lag projection; it does not implement MacKay evidence maximization, infer observation precision, or return posterior predictive uncertainty. No third-party implementation was inspected or copied.
