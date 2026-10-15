---
name: "Sonnet"
description: "Learnable wavelet atoms, spectral-coherence attention across variables, and a unitary Koopman operator over atoms, with a conv decoder. Use for multivariate data whose variables share frequency-domain dependencies; not for channel-independent data, level-shifting series (no instance norm), or probabilistic output."
---

# Sonnet

## Idea

- Naive Transformers struggle to model relationships among variables over time; Sonnet uses learnable wavelets, Koopman spectral dynamics, and Multivariable Coherence Attention (MVCA).
- `LearnableWavelets` builds `num_wavelets` damped-cosine atoms with learnable parameters and modulates the embedded series by each atom.
- `SpectralCoherence` (MVCA) scores variable pairs by FFT magnitude-squared coherence of their query and key and uses it as the attention weight.
- `StableKoopman` evolves the atom axis with a norm-preserving operator `U D U^H` (QR-orthonormalized, unit-modulus diagonal).
- Evolved atoms are recombined, passed through a small conv decoder, and `adaptive_avg_pool1d` resamples to `pred_len`; there is no instance normalization.

## When to use

- Multivariate data where variables are related through shared oscillatory components (spectral coherence), e.g. targets driven by other measured variables.
- Oscillatory dynamics that a stable linear (Koopman) operator over wavelet atoms can evolve.
- No instance normalization, so train/test level shifts are not compensated inside the model. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly (input embedding and decoder output width).
- Other hyperparameters: preset defaults in `configs/models/Sonnet.toml`; tune generically.

## Differences

Clean-room rewrite after inspecting `sonnet/mts_model/models/Sonnet.py` at `ClaudiaShu/Sonnet@bf3d480` (no license, `NOASSERTION`); nothing copied. Equations (1)-(3) map to `LearnableWavelets` and `SpectralCoherence`; `StableKoopman` builds `U diag(exp(i p)) U*`; reconstruction multiplies and sums evolved states by their atoms before the three-layer conv decoder.

- The paper separates one endogenous target from exogenous variables via an alpha-controlled joint embedding; TSFLab embeds all channels jointly and forecasts all of them.
- The decoder uses adaptive pooling for arbitrary horizons; dataset-specific target selection, alpha splits, and the original training/evaluation harness are not included.

Citation: Shu, Lampos, "Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting", AAAI 2026, doi:10.1609/aaai.v40i30.39736 (arXiv:2505.15312).
