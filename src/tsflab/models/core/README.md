---
name: "CoRe"
description: "Test-time adapter around a frozen forecaster: per-variate corrections interact through a shared-anchor low-rank bottleneck, gated by spectral entropy. Use for online adaptation of multivariate forecasts under distribution shift; not as a standalone forecaster or for running the streaming protocol itself."
---

# CoRe

## Idea

- Wraps a frozen forecaster (a last-value `ChannelWiseLinear` by default, or an external `base_forecast`); only the adapter, gates and bottleneck are adaptable (`adaptable_parameters`).
- A COSA-style base adapter gives per-variate corrections Delta; `refine` concatenates each with the cross-variate mean (anchor) and passes it through a rank-r tanh bottleneck, so variates interact in correction space.
- `spectral_descriptor` computes per-window spectral entropy and band-energy ratios; a tanh gate scales the refinement, starting closed.
- Final forecast: `Y_base + Delta + g * delta`.

## When to use

- Test-time adaptation of a deployed forecaster when the data drifts after training and labels arrive with delay.
- Cross-variate correction helps when channel errors are correlated.
- The streaming update loop is outside the model; without it the adapter starts as the base forecast.

## Configure

- `enc_in`: the dataset's channel count.
- `scr_rank`: bottleneck rank; `0` means `r = C`; the paper uses `r = 16` on high-dimensional datasets.

Other hyperparameters: preset defaults in `configs/models/CoRe.toml`; tune generically.

## Differences

- Independent rewrite from the paper; the official repository (no license) was a reference only.
- Streaming protocol (delayed-label buffer, PAAS, adaptive learning rate, Eq. 9 weight decay) is not run.
- The spectral descriptor is per sample (paper: per window; code: per batch); band edges follow the paper.
- Only the single-layer linear base adapter is implemented. Detail in reference.md.
