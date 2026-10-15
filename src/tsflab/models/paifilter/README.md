---
name: "PaiFilter"
description: "FilterNet's plain shaping filter: a learnable complex frequency mask on the rFFT of each channel, then a shared two-layer MLP, inside RevIN. Use for efficient long-horizon forecasting of noisy periodic series; not for cross-channel dependence, exogenous inputs, or probabilistic output."
---

# PaiFilter

## Idea

- `PlainShapingFilter` multiplies each channel's rFFT spectrum by one shared learnable complex weight per frequency bin (initialized to identity) and inverts it (paper Eq. 8).
- The filter selectively passes or attenuates frequency components, targeting high-frequency noise while using the whole spectrum.
- A two-layer MLP (GELU) maps the filtered window to the horizon, shared across channels; `revin` wraps the model.

## When to use

- Series whose structure lives in a few frequency components (strong or multiple periodicities) mixed with high-frequency noise.
- Tight compute budgets: one frequency mask plus an MLP, no attention.
- Not when cross-channel dependence carries the signal (channel-independent), when timestamps or covariates matter (marks are ignored), or when quantiles are needed (point output).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (sizes RevIN).
- The filter has `seq_len // 2 + 1` frequency weights and the MLP input is `seq_len`; both are derived from the task, not set.

Other hyperparameters: preset defaults in `configs/models/PaiFilter.toml`; tune generically.

## Differences

- Paper-driven local implementation of the universal complex kernel (Eq. 8); the Apache-2.0 official repository is reference-only and no source was copied.
- TSFLab reuses canonical RevIN.
- Only the plain shaping filter is implemented; the contextual shaping filter (TexFilter) is a separate model.
