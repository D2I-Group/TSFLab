---
name: "SparseTSF"
description: "Ultra-light (~1k parameters) cross-period sparse forecaster: fold the window by its period and forecast each phase series with one shared linear map. Use for strongly periodic series with a known period, long horizons, small data or tight compute; not for aperiodic series or cross-channel dependencies."
---

# SparseTSF

## Idea

- Cross-Period Sparse Forecasting decouples periodicity and trend: downsampling by the period turns the task into predicting cross-period trend per phase, with fewer than 1k parameters.
- A one-channel moving-aggregation `Conv1d` (kernel about `period`) is added residually, then the window is reshaped into `period` phase series over `history_periods` cycles.
- `forecaster` is one `nn.Linear(history_periods, future_periods)` (or a small MLP with `model_type="mlp"`) shared by all phases and channels.
- Outputs are interleaved back into time order, trimmed to `pred_len`, and the detached window mean is added back.

## When to use

- Series with a strong, stable period known in advance (e.g. daily cycle in hourly data); the model only extrapolates cross-period trend per phase.
- Long horizons, small samples, low-quality data, or tight compute: the paper reports good generalization with a tiny parameter count.
- Channel-independent with shared weights; no cross-channel modelling. Weak or drifting periodicity breaks the phase alignment. Point forecasts only.

## Configure

- `period` follows the dataset's dominant period in steps (e.g. 24 for hourly data with a daily cycle); `period <= seq_len`.
- `seq_len` and `pred_len` are best multiples of `period`: non-divisible histories use the last complete periods and non-divisible horizons are trimmed from a final partial period.
- `enc_in` follows the dataset channel count; it must equal it exactly.
- Other hyperparameters: preset defaults in `configs/models/SparseTSF.toml`; tune generically.

## Differences

Paper-driven local implementation; the official `lss-1138/SparseTSF@b8c2740` (Apache-2.0) is reference-only and no source was copied or adapted.

- Shared learnable period-aggregation convolution, mean-centered histories reshaped into phase-aligned subsequences, one shared Linear/MLP per sparse subsequence, phases interleaved back.
- Non-divisible histories and horizons are handled by using the last complete periods and trimming (local extension).

Citation: Lin, Lin, Wu, Chen, Yang, "SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters", ICML 2024 (PMLR 235), arXiv:2405.00946.
