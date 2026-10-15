---
name: "DLinear"
description: "LTSF-Linear baseline: moving-average trend/seasonal split with one linear map over time per component, summed. Use as a strong, cheap baseline for long-term forecasting of trend-plus-seasonal series; not for cross-channel dependence, covariates, or probabilistic output."
---

# DLinear

## Idea

- Splits the lookback into a moving-average trend and a seasonal remainder (`series_decomposition`).
- Forecasts each component with one linear map over time and sums them (`dlinear`); channels share weights unless `individual=True`.
- The reference showing that simple linear maps match or beat many Transformer forecasters on LTSF benchmarks.

## When to use

- First baseline for long-horizon forecasting, especially when trend and a stable seasonal pattern dominate.
- Tiny and fast; channels are univariate series, so cross-channel structure is ignored.
- Point output only; no calendar or covariate input.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/DLinear.toml`; tune generically (`kernel_size` odd).

## Differences

- Paper-driven local implementation reusing the catalog's `DLinearBackbone`, decomposition and channel-wise projection; the Apache-2.0 repository is reference only and no source was copied.
- A centered moving average separates trend and remainder; two temporal affine maps forecast the components.
- Configurable kernel and per-channel projections are local extensions.
