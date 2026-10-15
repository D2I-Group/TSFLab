---
name: "PGN"
description: "TPGN: folds each series by its period, runs a parallel gated network over rows (long-term) plus a pooled short-term branch, with per-variate linear heads. Use for long-range forecasting of periodic hourly data with calendar marks; not for aperiodic series or horizons that are not multiples of the period."
---

# PGN

## Idea

- `ParallelGatedNetwork` (the paper's PGN): a linear Historical Information Extraction layer summarizes every strictly earlier row in parallel (zero padded in front); one sigmoid gate mixes that summary with a tanh candidate, `Out = G*H + (1-G)*H^` (Eq. 1).
- `Model` is TPGN: the series is folded into `seq_len / period` rows of `period` columns, each step carrying the value plus four hourly calendar features; the long-term branch applies PGN along rows then pools the rows of each column.
- The short-term branch pools each row's columns to patches and the patches to one global vector repeated over the period.
- One linear head per variate maps `[global, long]` to the future rows, unfolded to `pred_len` (Eq. 5); every learned map has independent parameters per variate (`VariateLinear`). Instance normalization is `revin` without affine terms.

## When to use

- Long-range forecasting of series with a stable period, where the lookback spans several cycles: rows are aligned cycles, so the gated network models cycle-to-cycle evolution in parallel instead of step by step.
- Hourly data whose timestamps carry signal: calendar features enter every step (other frequencies need preprocessed four-wide marks; absent marks become zeros).
- Not for aperiodic series or when `seq_len`/`pred_len` cannot be multiples of the period; channels never interact (per-variate parameters).

## Configure

- `enc_in` follows the channel count: must equal the number of input channels (one parameter set per variate).
- `period` follows the dataset period: `seq_len` and `pred_len` must be multiples of `period`, and `seq_len` must cover at least two periods.

Other hyperparameters: preset defaults in `configs/models/PGN.toml`; tune generically.

## Differences

Independent rewrite from Sections 3.1-3.2 (Eqs. 1-5). The official `Water2sea/TPGN` (revision `4bbe2be`) has no license; it was read only to resolve layouts and omissions.

- `seq_len` and `pred_len` must be multiples of `period`; the official code rounds forecast rows up and keeps the trailing `pred_len` outputs, misaligning the horizon.
- Calendar features are the four hourly Time-Series-Library features only (official also accepts minutely and daily calendars).
- `revin` detaches the normalization statistics (official lets gradients flow through the standard deviation).
- `d_model=64`, `period=24` are presets, not the paper's per-dataset search; GRU/LSTM/MLP variants are not included, only the `use_short_branch` switch.
