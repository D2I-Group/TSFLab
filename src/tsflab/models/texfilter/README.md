---
name: "TexFilter"
description: "FilterNet contextual shaping filter: a learned complex map embeds each channel's spectrum, ReLU-gated complex gains build an input-dependent filter, then inverse FFT and an MLP. Use for channel-independent long-term forecasting of noisy, high-frequency-rich series; not for cross-channel structure or covariates."
---

# TexFilter

## Idea

- `ContextualShapingFilter` takes the rFFT of each channel and maps it with a learned complex matrix to an embedded spectrum (`embed_size // 2 + 1` bins).
- A data-dependent context is built by passing the embedded spectrum through two ReLU-activated complex gains; the embedded spectrum is multiplied by this context (Eq. 9).
- The filtered spectrum is inverse-FFT'd to an `embed_size` time axis and a two-layer GELU MLP maps it to the horizon, inside `revin`; all channels share weights.

## When to use

- Long-term forecasting where high-frequency noise hurts attention-based models and the whole spectrum should be used; the filter selectively passes or attenuates frequency components.
- Efficiency-sensitive settings: one complex linear map plus a small MLP per channel.
- Not when cross-channel interactions, exogenous covariates, or calendar effects matter (channel-independent, marks ignored).

## Configure

- `enc_in`: number of channels (sizes RevIN).

Other hyperparameters: preset defaults in `configs/models/TexFilter.toml`; tune generically.

## Differences

- Paper-driven local implementation of the contextual kernel of Eq. (9): a complex linear map embeds the input spectrum, complex learnable factors generate an input-dependent response, and the response filters the embedded spectrum before inverse FFT and forecasting.
- Only the contextual shaping filter is provided (the plain shaping filter variant is not part of this model); TSFLab reuses canonical RevIN.
- The external repository is reference-only; no source file was copied or adapted.
