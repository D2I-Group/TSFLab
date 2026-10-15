---
name: "QuantileDLinear"
description: "DLinear backbone (moving-average decomposition plus two linear maps) with a monotone quantile head trained by pinball loss. Use for cheap probabilistic baselines on trend-plus-seasonal data; not for complex nonlinear dynamics or when only point forecasts are needed."
---

# QuantileDLinear

## Idea

- `DLinearBackbone` produces a point forecast per channel from moving-average decomposition (`series_decomposition`) and two linear maps (trend and seasonal).
- `QuantileHead` expands each point into `Q` quantiles from a median anchor with cumulative softplus offsets, so quantiles cannot cross.
- `output_type = "quantile"` returns `(B, pred_len, C, Q)`; pair it with `[training] loss = "quantile"` (pinball). `quantile_levels` are injected from `evaluation.quantile_levels`.
- The head and pinball protocol are TSFLab additions, not part of the DLinear paper.

## When to use

- Tasks that need quantile forecasts at linear-model cost: a probabilistic baseline before heavier models.
- Data well described by a smooth trend plus a seasonal remainder, the DLinear assumption.
- Not for strongly nonlinear dynamics or cross-channel effects (channels share weights unless `individual`).

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/QuantileDLinear.toml`; tune generically.

## Differences

Independently composed from shared components; no official source copied.

- The monotone quantile head and pinball-loss protocol (Koenker and Bassett, 1978) are TSFLab additions; this is not a model or result claimed by the DLinear paper.

Cite: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (arXiv:2205.13504).
