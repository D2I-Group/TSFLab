---
name: "Amplifier"
description: "Adds the mirrored spectrum to expose low-energy frequencies, then semi-channel mixing, decomposition, restoration. Use for lightweight multivariate forecasting where weak frequency components carry signal next to dominant seasonality and trend; not for probabilistic or graph tasks."
---

# Amplifier

## Idea

- Energy amplification: adds the frequency-reversed one-sided FFT spectrum (`flipped_spectrum`) to the normalized input so low-energy components gain weight.
- `SemiChannelInteraction` splits a learned cross-channel commonality from channel-specific residuals and refines each along time.
- The amplified series is decomposed with `series_decomposition`; seasonal and trend parts are forecast by separate MLPs.
- `ComplexFrequencyProjection` restores the energy by subtracting the projected mirrored spectrum from the forecast spectrum; `revin` wraps the model.

## When to use

- Series whose spectrum is dominated by a few high-energy components while low-energy frequencies still matter for accuracy.
- Seasonal-trend structure: amplified data shows two spectral peaks that the separate seasonal and trend MLPs model independently.
- Lightweight MLP model for tight compute budgets; the SCI block adds mild channel commonality modelling (switch off with `sci = false`).
- Point output only.

## Configure

- `enc_in`: number of channels.
- `moving_average`: positive odd trend window of the decomposition.

Other hyperparameters: preset defaults in `configs/models/Amplifier.toml`; tune generically.

## Differences

Clean-room implementation; the Apache-2.0 author repository was used as reference only. Paper Eqs. 5-9 map to spectrum flip, amplification, complex horizon projection, and restoration; Eqs. 10-11 to the common/specific SCI paths; Eqs. 12-13 to shared seasonal-trend decomposition and two forecast MLPs. It uses one-sided real FFTs and separate real/imaginary restoration maps, exposes forecasting only, and makes no checkpoint, training-recipe, or published-metric comparison claim.
