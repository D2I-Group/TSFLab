---
name: "LassoRegressionTS"
description: "Shared linear lag-to-horizon map per channel with an L1 weight penalty, fit by gradient descent. Use as a sparse linear baseline for short training data or tight budgets; not for nonlinear dynamics, cross-channel effects, or level shifts (no normalization)."
---

# LassoRegressionTS

## Idea

- A single `nn.Linear(seq_len, pred_len)` is applied along time to every channel with shared coefficients.
- `aux_loss = l1_penalty * |W|_1` is exposed so the standard trainer adds the Lasso penalty to the forecasting loss.
- Fit by gradient descent rather than coordinate descent, so coefficients are only approximately sparse.

## When to use

- A sanity baseline when only a few lags carry signal; the L1 term shrinks uninformative lag weights.
- Short training windows or tight compute: one `seq_len x pred_len` matrix.
- Channels are treated independently with shared weights; no cross-channel information.
- Purely linear and without instance normalization, so nonlinear dynamics and level shifts are out of reach.

## Configure

- `enc_in`: number of data channels (input-shape check; coefficients are shared).

Other hyperparameters: preset defaults in `configs/models/LassoRegressionTS.toml`; tune generically.

## Differences

- Independent implementation of the Lasso objective; no external implementation inspected or copied.
- Direct multi-horizon lag projection optimized by gradient descent, not a coordinate-descent solver.
- Coefficients are shared across channels; the L1 term reaches the trainer as `aux_loss`.
- Citation: R. Tibshirani, "Regression Shrinkage and Selection via the Lasso", JRSS Series B 58(1):267-288, 1996, doi:10.1111/j.2517-6161.1996.tb02080.x.
