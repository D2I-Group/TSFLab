---
name: "WinNet"
description: "One-convolution forecaster: each channel's history is reshaped into a window-by-window grid (sub-windows by lag), decomposed by a 2D moving average, and weighed by a single 3x3 conv. Use for cheap forecasting of periodic channels whose period matches the window; not for cross-channel dependence or aperiodic data."
---

# WinNet

## Idea

- `sub_windows` (Eq. 1): a linear layer maps each channel's last-value-centered history to `window * window` points; each row is a sub-window of consecutive points (short-term variation) and each column a fixed-lag sequence (long-term variation).
- Dual heads (Eq. 2): the within-window tensor and its transpose (cross-window view) go through the same decomposition and convolution.
- `trend_pad` and `decompose_2d` (Eq. 3, Fig. 3): rows are padded with their neighbours in the flattened sequence; a `window x window` average gives the trend and the remainder is seasonal.
- `DecompositionCorrelationBlock` (Eq. 4): per channel, trend and seasonal are two input maps of one 3x3 `Conv2d` with a sigmoid, weighing their lagged correlation.
- `forward` (Eq. 5): both head outputs and the sub-window tensor are summed, flattened, and mapped to the horizon by one linear layer; the last value is added back.

## When to use

- Designed for periodic series: columns line up points one window apart, so the grid exposes periodic variation when `window` matches the period.
- Very small model (one convolution and two linear maps), suited to tight compute budgets.
- Channels are processed independently with per-channel kernels.
- Last-value centering, not RevIN, so scale changes are not normalized.

## Configure

- `enc_in`: must equal the dataset channel count (one 3x3 kernel per channel).
- `window`: grid side and sub-window length; it should follow the dominant period in steps (default 24 for hourly daily cycles). The grid has `window * window` points (576 by default).

Other hyperparameters: preset defaults in `configs/models/WinNet.toml`; tune generically.

## Differences

- Last-value normalization (official code) instead of the RevIN of Eq. (1).
- The DCB applies only the sigmoid after the convolution (code); Eq. (4) also lists a ReLU.
- Trend padding follows Fig. 3 for any `window`; the official 47 x 47 padding overwrites part of the tensor, uses an off-by-one wrap, and is hard-coded to 24 x 24.
- Both heads stack (trend, seasonal); the code swaps the order for the cross-window head and labels the moving average as seasonal.
- Per-channel kernels are one grouped convolution (same parameters and initialization fan-in).
