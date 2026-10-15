---
name: "SVTime"
description: "Small linear forecaster encoding inductive biases of large vision-model forecasters: patch-specific inter-period linear maps for seasonality plus a gated trend term on the backcast residual. Use for long-term forecasting of periodic series under tight compute; not for aperiodic data or cross-channel structure."
---

# SVTime

## Idea

- Reshapes the lookback into whole periods and applies a separate learned inter-period linear map per within-period patch (`PatchWisePeriodMap`), producing both a backcast and future periods.
- The backcast residual `history - backcast` is projected linearly as a trend and mixed with the period forecast by a learned scalar sigmoid gate.
- Channel-independent with shared weights; wrapped in optional reversible instance normalization (`revin`).

## When to use

- Long-term forecasting of series with a clear, known period (e.g. daily cycle in hourly data); the seasonal path works on whole periods.
- Resource-constrained settings: the paper reports rivaling large models with about 10^3 fewer parameters than vision-model forecasters.
- Lookbacks of several periods: only `seq_len // period` whole periods are used.
- Not for data without a stable period, or where cross-channel interactions matter.

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.
- `period`: the dataset's dominant seasonal period in steps (24 for hourly daily cycles); must be in `[1, seq_len]`; ideally `seq_len` is a multiple, since the remainder of the lookback is dropped.
- `patch_size`: within-period patch length, in `[1, period]`; ideally divides `period` (the last patch is shorter otherwise).

Other hyperparameters: preset defaults in `configs/models/SVTime.toml`; tune generically.

## Differences

- Clean-room rewrite from Sections 3.1, 3.2, 3.4 and Eq. 3 of the paper; no external implementation inspected or copied.
- Implements the named **SVTime** variant (learned patch-specific matrices encode IB1/IB2); the distance-attenuating annealing constraint of **SVTime-t** is deliberately not included.
- RevIN is a repository-side option; reported benchmark numbers and multi-block dataset tuning are not reproduced.
