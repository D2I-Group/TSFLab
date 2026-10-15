---
name: "Times2D"
description: "Multi-period 2D forecaster: the lookback is folded into cycle-by-phase grids for several periods, patched by strided Conv2D into residual-attention Transformers, plus a derivative heatmap branch for sharp changes. Use for long-term forecasting of multi-periodic series with abrupt turns; not for aperiodic data."
---

# Times2D

## Idea

- Periodic Decomposition Block (Eqs. 3-10): `period_grid` folds each channel into a grid per configured period (rows are cycles, columns phases); `PeriodBranch` turns full-height column blocks into tokens with a strided `Conv2d`, encodes them with a post-norm residual-attention Transformer, and maps back to `seq_len`; the period outputs are concatenated and projected to the horizon.
- First and Second Derivative Heatmaps (Eqs. 11-14): zero-left-padded first and second differences form `[B, N, T, 2]`; two `InceptionBlock2d` convolutions (channels as input maps) and two learned derivative weights (Eq. 16) give a second forecast.
- `forward` adds the two forecasts (Eq. 13) inside non-affine RevIN (`revin`).

## When to use

- Series with several strong periods (e.g. daily and weekly) whose intra- and inter-period variations a 2D view separates, plus sharp turning points the derivative branch highlights.
- Long lookbacks: the official ETTh1 setting uses `seq_len = 720` with periods up to 720.
- Periods must be chosen per dataset (fixed list, not detected at run time); not for aperiodic data. The heatmap branch mixes channels; the period branches are channel-independent.

## Configure

- `enc_in`: number of channels (inception input maps).
- `periods`: the dataset's periods in steps (e.g. from the training split's spectrum); defaults are the official ETTh1 values for `seq_len = 720`. Each period gives `ceil(seq_len / period)` grid rows.
- `patch_lens`: one patch width per period (same length as `periods`); columns are zero-padded to a multiple of it.

Other hyperparameters: preset defaults in `configs/models/Times2D.toml`; tune generically.

## Differences

- Independent rewrite from Section III (Eqs. 1-16) after reading the pinned official code (AGPLv3, recorded `AGPL-3.0-only`); nothing copied or imported.
- Fixed `periods`/`patch_lens` as in the official code, instead of the paper's per-batch top-k FFT periods (Eqs. 1-3).
- One shared learned derivative-weight pair (Eq. 16), instead of the official per-batch-row pair (and random per-forward weights for M4).
- Not implemented: unused official position-projection layers, the `add`, `wo_conv` and `serial_conv` flags, M4 short-term presets.
- Full detail: reference.md, `## Differences in detail`.
