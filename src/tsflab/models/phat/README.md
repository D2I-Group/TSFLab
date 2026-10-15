---
name: "PHAT"
description: "Period-heterogeneity Transformer: arranges each variable into a phase-by-cycle bucket from its own FFT period and applies positive-negative attention inside the bucket. Use for multivariate series whose variables have different or shifting periods; not for aperiodic data or tasks needing cross-variable mixing."
---

# PHAT

## Idea

- `_periods` takes the top FFT periods of each variable (or a fixed `period_list`); `_bucket_path` reshapes the embedded series into a phase-by-cycle bucket.
- `PositiveNegativeAttention` combines a positive softmax over phases closer than the query with a gated negative term over farther ones, plus an aligned same-phase attention, using distance masks (periodic alignment and deviation).
- Gated SiLU feed-forward blocks (`_PHATBlock`) follow the attention; the bucket is unfolded and projected to a scalar per step.
- The forecast is the mean of the bucket path (`forecast_projection`) and a plain `base_projection` linear path, inside `revin`.

## When to use

- Multivariate data where variables have distinct and changing periods (period heterogeneity): each variable is bucketed by its own detected period, avoiding interference from inconsistent periods.
- Seasonal data generally; the averaged linear path keeps a plain baseline when periodicity is weak.
- Not for tasks where cross-variable dependence carries the signal: interactions stay within a variable's bucket.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels (RevIN statistics).
- `period_list` (optional) follows the dataset period: fixed cycle lengths in steps, capped at `seq_len`; when absent, the top `period_topk` FFT periods of each window are used.

Other hyperparameters: preset defaults in `configs/models/PHAT.toml`; tune generically.

## Differences

Independent clean-room implementation of paper Eqs. (4)-(13); the MIT repository is kept as a pinned reference and no source was copied.

- Dataset-specific periods are not included; variates without periodicity (detected period not shorter than the lookback) form the zero bucket, unfolded and with absolute-distance attention.
- No comparison with published results is claimed.

Cite: Ma, Huang, Ma, Wang, Huang, Zhou, Wang, Wang, Wang, "PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting", ICLR 2026 (arXiv:2602.00654).
