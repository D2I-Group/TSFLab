---
name: "OLMA"
description: "Training loss, not an architecture: L1 on the channel-wise DFT, temporal DFT and Haar DWT of forecast vs label, attached here to a DLinear forecaster. Use for a cheap linear forecaster trained with a transform-domain loss; not for other backbones without re-attaching the objective, or probabilistic output."
---

# OLMA

## Idea

- OLMA changes the loss, not the architecture: the forecaster is the `dlinear` component (moving-average decomposition with seasonal and trend linear maps), and `forward` is its plain forecast.
- `Model.channel_loss` takes the DFT across channels at every horizon step (Eq. 11-12) and averages the absolute complex error; the paper motivates it as an entropy-reducing unitary transform of the labels.
- `Model.temporal_fourier_loss` and `Model.temporal_wavelet_loss` compare the temporal DFT and the single-level Haar DWT coefficients (`haar_dwt1d`) of every channel with L1 errors (Eq. 13-14).
- `Model.olma_loss` returns `loss_weight * (alpha L_c + beta L_tF + gamma L_tW)` (Eq. 15, defaults 0.34/0.33/0.33); `training_objective` uses it in place of the configured criterion.

## When to use

- Training a lightweight linear forecaster with a loss that compares forecasts in frequency and wavelet domains instead of point-wise MSE.
- Multivariate targets: the channel-DFT term couples channels in the loss even though the DLinear carrier is channel-independent.
- Not with `features="MS"` if the channel term matters (it degenerates to a single channel), not for probabilistic output (point forecast only), and not as a drop-in for other backbones (only the DLinear carrier is provided).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `weight_channel`: weight of the channel-DFT term; default 0.34 (with 0.33/0.33), while the paper uses 0.1 with 0.45/0.45 on the many-channel ECL and Traffic sets. The sum to 1 is not enforced.

Other hyperparameters: preset defaults in `configs/models/OLMA.toml`; tune generically.

## Differences

- Independent implementation; the MIT official repository ships only `OLMA.py` (the loss, no training pipelines) and nothing was copied.
- One carrier, DLinear (shared projections, kernel 25); the paper also applies OLMA to seven other backbones.
- Each term is a mean absolute error (official code) rather than the paper's summed L1, and the channel and temporal DFTs use the real FFT, as in the code.
- Only the Haar wavelet is provided; odd horizons are replicate-padded instead of zero-padded, which leaves the loss unchanged.
- OLMA replaces the time-domain loss; the OLMA + MSE variant of Appendix A.4 is not provided. Full detail in reference.md.
