---
name: "WDAN"
description: "Wavelet-disentangled point-level normalization around PatchTST: wavelet trend as per-step mean, residual sliding std as per-step scale, and an MLP predicting future statistics. Use for non-stationary series with drifting trend and changing variance; not for stationary data or cross-channel modelling."
---

# WDAN

## Idea

- `WaveletTrend` (Eqs. 1-3): a K-level DWT with fixed orthonormal filters (coiflet-3 default) and half-sample symmetric extension; the trend is the inverse of the level-K approximation and the residual is `x - trend`.
- `normalize` (Eqs. 4-6): every time step is standardized by its own mean (the trend) and scale (`sliding_std` over the residual) before the `patchtst` backbone (its RevIN disabled).
- `StatisticsPredictor` (Eqs. 7-12): linear embeddings of the centered trend, its difference, the residual, and the centered scale predict the future mean and scale series; `denormalize` applies Eq. (14).
- Three stages (Eqs. 15-17): `pretrain` fits and freezes the statistics predictor on the horizon's own statistics, the backbone trains alone for `joint_after_epochs` epochs, then both train jointly (`advance_stage`).

## When to use

- Designed for non-stationary series whose level and variance drift within and across windows; instance-level normalization would use one statistic per window.
- Point-level statistics follow a smooth trend and local volatility, so strong trends and heteroscedastic noise are the target conditions.
- Channels are handled independently and marks are ignored.
- The statistics pretraining stage adds training cost.

## Configure

- `enc_in`: must equal the dataset channel count.
- `patch_len`, `stride`: PatchTST patching of `seq_len` with end padding (one extra patch); `patch_len <= seq_len`.
- `dwt_levels`: wavelet levels for the trend; more levels give a smoother trend and need a longer `seq_len` (official default 1).

Other hyperparameters: preset defaults in `configs/models/WDAN.toml`; tune generically.

## Differences

- The backbone is fixed to the catalog `patchtst` component; the paper also evaluates iTransformer, Crossformer, and FEDformer.
- The sliding standard deviation is the population form of Eq. (5); the code uses the unbiased estimator.
- Only `haar`, `db4`, and `coif3` filters are tabulated.
- The statistics stage runs a fixed number of epochs without the official best-validation snapshot and learning-rate decay; the stage-2/3 switch counts training batches.
