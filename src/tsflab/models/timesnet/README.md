---
name: "TimesNet"
description: "Task-general 2D-variation backbone: the embedded series is folded into 2D images by its FFT-detected dominant periods and processed by inception 2D convs, aggregated by period amplitude. Use for multi-periodic series with calendar marks; not for aperiodic data or very wide channel sets at the default width."
---

# TimesNet

## Idea

- The lookback is embedded with values plus six-column calendar marks (`forecast_embedding`) and extended to `seq_len + pred_len` by a linear map on the time axis; the forecast is the last `pred_len` steps.
- `dominant_periods` picks the top-k FFT periods of the hidden series; `TimesBlock` pads and folds it into (cycles, period) images, so rows carry inter-period and columns intra-period variation.
- The cataloged `inception_block` (parallel same-padded 2D kernels, averaged) is applied in a two-layer conv stack per image; outputs are unfolded and summed with softmax amplitude weights plus a residual.
- Inputs are instance-standardized per channel and restored at the output.

## When to use

- Series with multiple periodicities whose intra- and inter-period variations are informative (e.g. daily and weekly cycles), in long- or short-term forecasting.
- Data with calendar effects: the six-column marks enter the embedding.
- Channels are mixed only through the embedding and output projection; cost grows with `d_model` and `seq_len + pred_len`. Not for aperiodic, noise-like series.

## Configure

- `enc_in`, `c_out`: number of channels; must be equal.
- `top_k`: dominant periods per block; at most `(seq_len + pred_len) // 2`.

Other hyperparameters: preset defaults in `configs/models/TimesNet.toml`; tune generically.
The official scripts (pinned revision) use `d_model`/`d_ff` of 16-64 for data with up to 21 channels
(ETT, Weather, Exchange) and 256-768 for ECL, Traffic, and ILI; the preset follows the former:
512/2048 trained two orders of magnitude slower per epoch on 8-channel hourly data (2026-10-09).

## Differences

- Clean-room implementation; the MIT THUML codebase is reference-only and was not copied.
- Covers FFT top-frequency selection, period-aligned 1D-to-2D reshaping, inception convolution, softmax amplitude aggregation and residual TimesBlocks; non-forecast tasks, official recipes, and checkpoint or published-metric comparisons are omitted.
- Detected periods are clamped to at least 1 (the official division is unguarded, so an all-zero spectrum can give period 0).
- The inception kernels keep PyTorch's default `Conv2d` initialization (`init_weight=False`); the official `Inception_Block_V1` uses Kaiming-normal fan-out weights and zero bias.
