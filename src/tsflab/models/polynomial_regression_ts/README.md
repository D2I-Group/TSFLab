---
name: "PolynomialRegressionTS"
description: "Classical baseline: integer powers of the lag window feed one linear map to the horizon, shared by all channels. Use as a cheap nonlinear reference next to linear baselines, especially on short training sets; not as a competitive model for complex or cross-channel dynamics."
---

# PolynomialRegressionTS

## Idea

- `polynomial_features` concatenates element-wise powers of the lag window (default degree 2) along the time axis.
- One `nn.Linear(seq_len * degree, pred_len)` is shared by every channel; there is no normalization or nonlinearity beyond the powers.
- Trained end-to-end with gradient descent in the standard loop (a zero `aux_loss` is exposed).

## When to use

- A sanity-check baseline: it shows how much simple nonlinear lag features add over a purely linear map, at negligible compute.
- Small capacity suits short training windows.
- Not for cross-channel dynamics (channels share weights and never interact) or complex seasonal structure; without normalization, high powers of unscaled data can be unstable, so keep data standardized.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/PolynomialRegressionTS.toml`; tune generically.

## Differences

Independent implementation from the classical polynomial-regression description; no external implementation was inspected or copied.

- Integer powers of each lag only, without cross-lag or cross-channel interaction monomials.
- A direct multi-horizon map learned by gradient descent rather than a closed-form least-squares fit.

Cite: Norman R. Draper and Harry Smith, Applied Regression Analysis, 3rd ed., Wiley, New York, 1998, doi:10.1002/9781118625590.
