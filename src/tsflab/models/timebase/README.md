---
name: "TimeBase"
description: "Ultra-lightweight segment-level forecaster: folds the lookback into period-length segments, compresses them to a few basis components and maps those to future segments. Use for long-term forecasting of periodic, low-rank series under tight compute; not for aperiodic data or cross-channel structure."
---

# TimeBase

## Idea

- Folds the lookback into a `period_len` x segment grid, compresses the segment axis to `basis_num` basis components with `ts2basis`, then expands to future segments with `basis2ts` (segment-level, low-rank forecasting, Eqs. 1-4).
- Subtracts a per-period or per-series mean before the maps and adds it back afterwards (`use_period_norm`).
- During training `aux_loss` adds `orthogonal_weight` times the off-diagonal Frobenius norm of `G = X_basis^T X_basis` (Eqs. 5-7) to keep bases decorrelated.
- Weights are shared across channels unless `individual=True` creates a pair per channel.

## When to use

- Long-term forecasting of series with strong periodic similarity and low-rank structure across periods, which the paper argues is common at long horizons.
- Very tight compute or memory budgets: the model is two small linear maps over segments.
- Not for series without a stable period, with dominant cross-channel effects, or needing covariates.

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.
- `period_len`: the dataset's dominant period in steps (24 for daily cycles in hourly data), in `[1, seq_len]`. Ideally `seq_len` and `pred_len` are multiples; otherwise the lookback is padded with an earlier segment and the forecast is trimmed.

Other hyperparameters: preset defaults in `configs/models/TimeBase.toml`; tune generically.

## Differences

- Checked against the pinned `models/TimeBase.py`; the local module was written for TSFLab and copies no external source.
- `orthogonal_weight = 0.08` is a runnable point from the paper's 0.00-0.20 sweep, not a universal paper setting; tune per dataset.
- Dataset-specific result reproduction is not claimed.
