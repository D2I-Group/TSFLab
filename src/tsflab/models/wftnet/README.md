---
name: "WFTNet"
description: "TimesNet-style forecaster that combines FFT-period 2D maps (global periodicity) and Morlet wavelet time-scale maps (local periodicity) through shared Inception convolutions, weighted by a dataset periodicity coefficient. Use for long-term forecasting of periodic series; not for aperiodic data or probabilistic output."
---

# WFTNet

## Idea

- Global branch (Eq. 1): the cataloged `dominant_periods` picks the top-k FFT periods; each folds the embedded sequence into a 2D map for the Inception block, summed with amplitude-softmax weights.
- Local branch (Eq. 2): a real Morlet CWT at 8 scales `2 ** linspace(-1, wavelet_scale, 8)` maps each embedding channel to a scale-by-time image, tabulated once per length.
- One `inception_block` is shared by both branches; a depthwise strip convolution of height 8 compresses the scale axis.
- The Periodicity-Weighted Coefficient `alpha` (`period_coeff`, Section 3.4) mixes them: `alpha^n X_f + (1 - alpha^n) X_w + X` with `n = pwc_power`.
- Frame: non-affine RevIN, value + position + hourly calendar embedding, a linear map from `T_s` to `T_s + T_p` steps, stacked blocks with one shared LayerNorm, projection to the variates, and the last `T_p` steps.

## When to use

- Designed for long-term forecasting of series with periodic structure: strong global periods favour the Fourier branch, while local, time-varying periodicity favours the wavelet branch.
- `alpha` is the share of the strongest of the lowest FFT bins, so a spectrum concentrated in one bin weights the Fourier branch; a spread spectrum weights the wavelet branch.
- Calendar marks enter the embedding as hourly features.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count.
- `period_coeff`: required; the Section 3.4 statistic (`m = 5` lowest FFT bins including DC, raw values, all numeric columns) computed on the training split only (`periodicity_weighted_coefficient` in `model.py`). Each dataset or split needs its own value; the preset 0.8126 is for ETTh2. `0` gives the Wavelet-Only ablation.
- `top_k`: number of FFT periods folded per block; must be `>= 1`.

Other hyperparameters: preset defaults in `configs/models/WFTNet.toml`; tune generically.

## Differences

- `alpha` is a model parameter computed on the training split; the official code computes it from the whole CSV, test period included (0.7845 for ETTh2).
- The official `top_k = 0` wavelet-only block (ETTh1, ETTm2, Weather, some ETTm1 scripts) is not the paper's WFTBlock and is not implemented.
- The CWT is a fixed tabulated `[8, T, T]` operator in float64, not recomputed by FFT each call (same transform up to rounding).
- Marks are always the four hourly `timeF` features derived from the raw marks (the official minute-level datasets use five); the unused `ms_wave` option is not implemented.
