---
name: "FreqMoE"
description: "Band-wise frequency mixture of experts denoises the window; residual complex frequency-extension blocks then forecast in the spectrum. Use for channel-independent forecasting of series whose structure separates into frequency bands; not for cross-channel modelling."
---

# FreqMoE

## Idea

- `FrequencyBandMixtureOfExperts` (`freq_band_moe`) reconstructs a denoised series from contiguous frequency bands combined by an input-dependent gate.
- Each `FrequencyExtensionBlock` standardizes the series, upsamples the rFFT spectrum from `seq_len` to `seq_len + pred_len` bins with a complex `nn.Linear`, applies `ComplexReLU`/`ComplexDropout`, refines with a second complex linear, and inverts.
- Blocks are chained on the backcast residual and their forecast segments are summed.

## When to use

- Designed for series whose components live in different frequency bands, letting experts weight bands per input before forecasting.
- Channel-independent with shared weights; no cross-channel interaction.

## Configure

- `enc_in`: number of channels.

Other hyperparameters: preset defaults in `configs/models/FreqMoE.toml`; tune generically.

## Differences

Independent rewrite from the paper; `models/FreqMoE.py` (`FreqDecompMoE`, `Block`, `ComplexReLU`, `ComplexDropout`, `Model`) of `sunbus100/FreqMoE-main` at `b34e9370` (no license file, recorded `NOASSERTION`) was consulted for details only, nothing copied.

- The paper says band boundaries are learned end to end, but the official code casts them to integer indices, which blocks autograd. This entry matches the official behaviour: `band_boundaries` is a fixed buffer under the former state-dict key, so older checkpoints load with unchanged outputs.
- The shared component's straight-through `learnable_boundaries=True` exists but is not enabled here.
- `ComplexReLU` and `ComplexDropout` stay model-local paper-specific glue.

Citation: "FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts." AISTATS 2025. arXiv:2501.15125.
