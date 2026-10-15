---
name: "RidgeRegressionTS"
description: "Classical baseline: one lag-to-horizon linear map shared by all channels, with an L2 weight penalty added to the training loss (gradient-trained ridge). Use as a regularized linear reference, especially with short training sets; not for nonlinear, cross-channel, or level-shifting data."
---

# RidgeRegressionTS

## Idea

- A single `nn.Linear(seq_len, pred_len)` is applied over time and shared by all channels, with no normalization.
- `forward` sets `aux_loss = l2_penalty * ||W||^2` (default 1e-4), which the standard trainer adds to the task loss, making the fit a gradient-trained ridge regression.
- The L2 penalty shrinks large coefficients toward zero, improving generalization when lags are many or correlated.

## When to use

- A regularized linear baseline: shows how far a shrunk direct linear map goes before any deep model.
- Short training sets, where the L2 penalty limits overfitting of the `seq_len x pred_len` weights.
- Not for nonlinear dynamics or cross-channel effects; there is no instance normalization, so level shifts between splits are not absorbed (see RLinear for the normalized variant).

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/RidgeRegressionTS.toml`; tune generically.

## Differences

Independent implementation from the cited ridge objective; no external implementation was inspected or copied.

- A direct multi-horizon lag projection optimized by gradient descent rather than the closed-form ridge estimator.
- Coefficients are shared across channels; the L2 term is exposed to the trainer as `aux_loss`.

Cite: Arthur E. Hoerl and Robert W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems", Technometrics 12(1):55-67, 1970, doi:10.1080/00401706.1970.10488634.
