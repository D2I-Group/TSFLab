---
name: "RLinear"
description: "Linear baseline: reversible instance normalization around one channel-shared affine map from lookback to horizon. Use as the reference linear model for long-term forecasting of periodic or trending series; not for non-periodic signals or channels with differing periods."
---

# RLinear

## Idea

- `revin` normalizes each instance and de-normalizes the forecast (`affine` and `subtract_last` options).
- `channel_wise_linear` is one `seq_len` to `pred_len` affine map `Y = XW + b`, per channel only if `individual=True`.
- Optional input dropout (`dropout`) is the only regularizer; no decomposition or nonlinearity.

## When to use

- The paper finds affine mapping dominates common long-term benchmarks and captures periodic patterns well; RevIN turns non-periodic trends into periodic-like patterns.
- A cheap, strong baseline for any long-term forecasting comparison; longer lookbacks help on multi-channel data with different periods.
- Struggles with non-periodic signals and with channels that have different periods (shared map); no cross-channel modelling.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/RLinear.toml`; tune generically.

## Differences

Written for TSFLab from Section 2.3 of the paper (`Y = XW + b`) after inspecting `models/RLinear.py` of the unlicensed reference repository at the pinned revision; no source copied.

- Default: parameter-free RevIN followed by one channel-independent affine map, as in the paper baseline.
- TSFLab additionally exposes channel-specific maps, affine/subtract-last RevIN variants, and input dropout as explicit ablations.
- No benchmark-training reference comparison is claimed.

Cite: Zhe Li, Shiyi Qi, Yiduo Li, Zenglin Xu, "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping", arXiv:2305.10721 (2023).
