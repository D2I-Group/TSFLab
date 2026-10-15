---
name: "FITS"
description: "Low-pass rFFT, one learned complex frequency interpolation to the extended length, then irFFT; about 10k parameters. Use for lightweight forecasting of smooth, periodic series under tight compute; not for series whose signal lives in high frequencies or needs cross-channel modelling."
---

# FITS

## Idea

- Keeps only the first `cut_freq` rFFT bins of the standardized window (low-pass filter discarding high-frequency components).
- `ComplexFrequencyInterpolation` is one complex affine map from retained input bins to the bins of the `seq_len + pred_len` spectrum, shared across channels unless `individual = true`.
- Zero-pads the spectrum, applies irFFT over the full length, rescales by `(seq_len + pred_len) / seq_len`, and keeps the last `pred_len` steps.
- Normalization is a detached per-window mean/std, restored on the output.

## When to use

- Designed for compact models (edge deployment): comparable accuracy with around 10k parameters on series dominated by low-frequency periodic structure.
- The low-pass cut removes spikes and fast fluctuations, so it underfits noisy or high-frequency signals.
- Channel-independent; no cross-channel interaction.

## Configure

- `enc_in`: number of channels (also the number of per-channel maps when `individual = true`).
- `cut_freq`: number of retained bins, capped at `seq_len // 2 + 1`; it should include the dominant period's bin, `seq_len / period`, and ideally some harmonics.

Other hyperparameters: preset defaults in `configs/models/FITS.toml`; tune generically.

## Differences

Paper-driven local implementation; `VEWOXIC/FITS` at `d040bb01` (Apache-2.0) is reference only, nothing copied.

- Implements reversible instance centering/scaling, rFFT low-pass truncation, complex frequency interpolation, zero padding, irFFT, length-ratio energy compensation and inverse normalization.
- Returns only the forecast horizon; backcast/reconstruction supervision and anomaly detection are not claimed.

Citation: Xu, Z., Zeng, A., Xu, Q. "FITS: Modeling Time Series with 10k Parameters." ICLR 2024. arXiv:2307.03756.
