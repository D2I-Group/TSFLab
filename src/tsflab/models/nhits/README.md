---
name: "NHiTS"
description: "Channel-independent MLP stacks with multi-rate input pooling and interpolated coarse forecasts, one frequency band per stack. Use for long-horizon forecasting of smooth multi-scale signals under tight compute; not for cross-channel interaction, exogenous inputs, or probabilistic output."
---

# NHiTS

## Idea

- `NHiTSBlock` pools its input with `n_pool_kernel_size` (max or average pooling) before an MLP, so each stack sees a different input rate.
- Each block predicts only `ceil(pred_len / n_freq_downsample)` forecast knots and expands them to the horizon by interpolation (`interpolation_mode`).
- Blocks are chained with backward residuals (input minus a full-window backcast) and forward summation, starting from a last-value forecast.
- `revin` (no affine) wraps the model; every channel is folded into the batch and shares the blocks.

## When to use

- Long-horizon forecasting where the target is smooth enough that coarse knots plus interpolation approximate the horizon (the paper's motivation: volatility and cost of long horizons).
- Signals that mix components at different frequencies and scales; stacks with coarse-to-fine pooling and downsampling split them.
- Tight compute budgets: an MLP-only model, reported an order of magnitude faster than Transformer baselines.
- Not when cross-channel interaction carries the signal (channels are independent), when exogenous covariates or calendar marks matter (marks are ignored), or when quantiles are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `n_freq_downsample`: per-stack factors that follow `pred_len`; each block emits `ceil(pred_len / factor)` knots, so keep factors well below `pred_len`. List length must match `stack_types`.
- `n_pool_kernel_size`: per-stack pooling kernels that follow `seq_len`; the pooled input has `ceil(seq_len / kernel)` steps. List length must match `stack_types`.

Other hyperparameters: preset defaults in `configs/models/NHiTS.toml`; tune generically.

## Differences

- Rewritten for TSFLab after checking the paper and the pinned official implementation (`neuralforecast`); no source file copied or mechanically adapted.
- Blocks regress a full-window backcast and coarse forecast knots; only the forecast knots are interpolated. Forecasts start from the last observed level.
- Only the paper's `identity` interpolation basis is supported; canonical RevIN is optional (`use_norm`).
- The runnable default is not a paper benchmark reproduction.
