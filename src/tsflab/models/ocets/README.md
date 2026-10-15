---
name: "OCETS"
description: "DLinear backbone that predicts a per-step softmax over value bins, trained with ordinal cross-entropy against truncated-Gaussian soft labels; the point forecast is the expected bin value. Use for cheap forecasting with a categorical predictive distribution; not for series that leave the lookback's value range."
---

# OCETS

## Idea

- `truncated_gaussian_bin_probabilities` is the Target-to-Probability Transformation (Eqs. 8-11): a Gaussian of width `sigma` centered at the scaled target, truncated to the support and integrated over `num_bins` equal-width bins.
- `Model.bin_probabilities` is the ordinal classifier: the shared `dlinear` backbone maps the min-max scaled lookback to `pred_len * num_bins` logits per channel, softmaxed over bins as `[batch, horizon, channel, bin]`.
- `ordinal_cross_entropy` is Eq. (2): binary cross-entropy between true and predicted cumulative distributions at the `K - 1` cut points (Eq. 15); `training_objective` uses it in place of the configured loss.
- `forward` returns the probability-weighted bin centers (Eq. 18), denormalized with the lookback min/max (Eqs. 74-77); `predict_distribution` exposes the full categorical forecast on the input scale.

## When to use

- Forecasting where a per-step predictive distribution over values is wanted next to a point forecast, at DLinear cost.
- Series whose future stays within the range seen in the lookback window: bins span the lookback's min-max range.
- Not for series that trend or jump beyond the lookback's min and max (the forecast is confined to that range), when cross-channel interaction or exogenous/calendar inputs matter (marks are ignored), or when the catalog's quantile output is required (the declared output is a point forecast).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/OCETS.toml`; tune generically.

## Differences

- Independent rewrite from the paper (Eqs. 2, 8-18, 74-77) after reading the pinned MIT official code; nothing copied or imported.
- Head layout, bin edges and centers, loss reduction, clamps and defaults (`num_bins = 45`, `sigma = 0.015`, kernel 25) follow the official code where the paper is inconsistent or silent.
- Predictions are denormalized with the lookback min/max (paper appendix). The official test loop uses the future window's min/max (leakage), so official reported numbers are not comparable.
- Validation, early stopping and test metrics use the point forecast; the categorical distribution is not a catalog `output_type`. `training.loss` is ignored.
- Alternative truncated t and Laplace labels and the NLinear/Linear backbones are not implemented. Full detail in reference.md.
