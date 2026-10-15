---
name: "DropoutTS"
description: "Sample-adaptive dropout: a spectral-residual noise score sets per-sample dropout rates at every PatchTST dropout site (straight-through). Use for training on noisy windows whose noise level varies between samples; not for clean data or as a change to the inference-time forecaster."
---

# DropoutTS

## Idea

- `SpectralNoiseScorer` detrends each channel by least squares, takes the rFFT, min-max normalizes the log-amplitude, and soft-masks bins below an SFM-anchored threshold `sigmoid(w_s * SFM + b_s)` (Eq. 2-6).
- The noise score is the MAE between the window and its masked reconstruction plus trend; within-batch min-max scores map to `p_min + (p_max - p_min) * tanh(s * Softplus(gamma))` (Eq. 7-9).
- Every `nn.Dropout` of the `patchtst` backbone becomes `SampleAdaptiveDropout` with a per-sample Bernoulli and a straight-through term, so the task loss alone trains the scorer (Eq. 10-11).
- The scorer runs only in training; evaluation is the plain PatchTST forecast.

## When to use

- Designed for robust forecasting when training windows carry different, unknown noise levels: noisier samples get stronger dropout.
- A training-time regularizer only; inference cost and behaviour equal PatchTST.
- The paper is backbone-agnostic; this entry fixes PatchTST (channel-independent).

## Configure

- `enc_in`: number of channels (the SFM scale and bias are learned per channel).
- `p_min < p_max` (validated); the paper tunes only the sensitivity (official grid 1, 5, 10).

Other hyperparameters: preset defaults in `configs/models/DropoutTS.toml`; tune generically.

## Differences

Independent implementation after reading `CityMind-Lab/DropoutTS` at `64a096ec` (Apache-2.0); nothing copied. `forward` reads only `x_enc`.

- Backbone: the catalog `patchtst` component sized after BasicTS PatchTST defaults, not the BasicTS code; its functional attention dropout is not adapted (as officially).
- SFM uses the official power of the `log1p` amplitude, with per-channel scale and bias, not the paper's scalar form on `A`.
- Range clamp `1e-6` and tie score 0.5 follow the code; defaults `p_min = 0.05`, `init_sensitivity = 1` follow the paper.
- A non-divisible batch shape raises instead of the official mean-rate fallback. Details in `reference.md`.
