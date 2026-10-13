---
name: "RhyMix"
description: "Dual-path hourly forecasting with learned multi-period tables, depthwise temporal convolutions and adaptive four-head fusion. Use for seasonal multivariate data; fixed hourly periods are unsuitable for other sampling frequencies."
---

# RhyMix

## Idea

- Normalize each lookback, mix channels through a residual bottleneck, and compute normalized five-statistic gate features.
- Combine tables with periods 12/24/48/168 and a two-block dilated temporal convolution path.
- Both paths apply direct, trend/seasonal, local residual convolution and periodic residual fusion heads. Future cyclic priors enter the periodic head before the learned head gate.
- Softmax head and hybrid gates fuse channel-resolved predictions before inverse normalization.

## When to use

- Hourly seasonal multivariate forecasting with lookback at least 13 and all input channels available.
- Not for arbitrary sampling frequencies, exogenous covariates or probabilistic forecasts.

## Configure

- `enc_in` must equal the input channel count. Periods are fixed to the hourly paper setup.
- Paper training uses Huber delta1, AdamW, cosine annealing to a nonzero minimum, clipping and validation early stopping; configure these explicitly using native training controls.

## Differences

- Independent implementation checked against the v1 equations, with no author-code parity claim. CNN layout and head parameter sharing remain declared interpretations.
- All gates consume normalized statistics; local and periodic residuals and channel identity are retained. Earlier archived implementations are not silently replaced.
- Paper split-independent windows require `task.window_context="within-split"`; the default runner carries prior-split context.
- The compact equation map and remaining assumptions are in [reference.md](reference.md).
