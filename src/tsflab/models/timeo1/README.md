---
name: "TimeO1"
description: "Time-o1 training objective: forecasts and labels are projected onto a per-variate SVD basis of the training labels and the leading decorrelated components are aligned, carried by a small MLP-plus-linear forecaster. Use for long horizons with strongly autocorrelated labels; not as an architecture study."
---

# TimeO1

## Idea

- `fit_projection` computes a per-variate SVD basis from standardized training labels (`training_setup` runs it once on the training split); `transform` projects forecasts and targets onto it.
- `transformed_alignment_loss` mixes an L1 loss on the top `rank_ratio` components with the temporal term via `alpha` (Eq. 5).
- The forecaster is only a runnable carrier: a channel-independent MLP plus linear skip; `training_objective` combines the transformed-label term with the configured criterion.

## When to use

- Long-horizon forecasting where temporal MSE suffers from label autocorrelation and a large number of per-step tasks; the transform yields decorrelated components ranked by significance.
- When studying the objective: Time-o1 is compatible with various forecasters, and here it ships with a deliberately simple carrier.
- Not for comparing architectures, or for cross-channel or covariate-driven targets (the carrier is channel-independent and ignores marks).

## Configure

- `enc_in`: number of channels; one `pred_len x pred_len` basis per channel, fitted from the training labels (for `MS` targets only the trailing channel).

Other hyperparameters: preset defaults in `configs/models/TimeO1.toml`; tune generically.

## Differences

- Checked against the pinned `utils/polynomial.py`, `README.md` and the official trainer; the local module was written for TSFLab and copies no external source.
- The runner trains with `alpha * mean|transformed difference| + (1 - alpha) * criterion(forecast, target)`: mean reductions and the configured criterion follow the official trainer (which also precomputes the basis from training data); `transformed_alignment_loss` keeps the paper's summed Eq. (5) as its default.
- The forecasting architecture is a local carrier; Time-o1 does not prescribe one.
