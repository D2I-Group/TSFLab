---
name: "EvoMSN"
description: "Multi-scale slice normalization at the top-k FFT periods, a shared DLinear backbone, predicted future slice statistics and an amplitude-weighted scale ensemble. Use for forecasting under distribution shift with several periodic scales; not for the paper's online setting or when a non-DLinear backbone is wanted."
---

# EvoMSN

## Idea

- Period search (Eq. 1): rFFT amplitude averaged over training windows and channels, too-long periods excluded, top-`k` frequencies mapped to integer periods `seq_len // f`.
- For each period the window is cut into period-length slices and standardized per slice, giving `k` normalized copies that one shared `dlinear` backbone forecasts (Eqs. 2, 6-7).
- `ScaleStatisticsPredictor` (Eq. 5): per period, MLPs over slice statistics and the raw window predict future slice means and standard deviations; outputs are denormalized slice by slice (Eq. 8).
- Each scale's output is weighted per sample and channel by the window's amplitude at that period (Eqs. 9-10).
- `pretrain` (offline Section 3.3) selects periods and fits the statistics predictors to true future slice statistics, freezes them, then the trainer fits the backbone.

## When to use

- Designed for non-stationary series whose mean and scale drift within and across windows, where statistics differ by time scale.
- Uses several periods at once, so it suits multi-periodic data; periods come from the training spectrum.
- Only the offline two-stage variant runs here; the online alternating updates of the paper need streaming test-time updates.

## Configure

- `enc_in`: number of channels.
- `periods`: selected by `pretrain` from the training-split spectrum (`top_k` of them); pass them explicitly to fix them, and load checkpoints into a model with the same periods.

Other hyperparameters: preset defaults in `configs/models/EvoMSN.toml`; tune generically.

## Differences

Independent rewrite of Sections 3.1-3.3 (Eqs. 1-11) after reading `qindalin/EvoMSN` at `2278adeb` (Apache-2.0); nothing copied.

- Offline two-stage training only; the backbone is fixed to DLinear (one of the paper's five).
- Slice standard deviations use the paper's population form (the code uses the unbiased estimator).
- Ensemble amplitudes come from the input window as Eq. (9) states (the code reads the last scale's backbone output).
- The statistics stage runs fixed epochs without the official early stopping and learning-rate schedule.
- Before `pretrain`, the `top_k` longest admissible periods serve as a data-free placeholder. Details in `reference.md`.
