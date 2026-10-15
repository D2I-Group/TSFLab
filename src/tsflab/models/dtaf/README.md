---
name: "DTAF"
description: "Dual-branch patch Transformer for non-stationary series: an MoE filter subtracts temporal non-stationary patterns and a frequency-differencing branch keeps shifting spectral bins. Use for long-term forecasting under distribution and spectral shift; not for stationary data or cross-channel modelling."
---

# DTAF

## Idea

- `TemporalStabilizingFusion` (TFS) routes each patch token over small expert MLPs and subtracts the weighted output as non-stationary content, then fuses causal history with a gated current token.
- `FrequencyWaveModeling` (FWM) keeps only the DC bin and the `top_k` bins with the largest adjacent-bin amplitude change (frequency differencing), reconstructs the series, and applies self-attention.
- `DTAFBlock` concatenates and fuses the two branches residually.
- Channels share weights through repeat-padded patch tokens; a flatten linear head forecasts; the input is standardized per window.

## When to use

- Designed for long-term forecasting of non-stationary series with temporal distribution shift and spectral variability (energy, finance, transportation, cloud workloads in the paper).
- Channel-independent: no cross-channel interaction.
- Little benefit expected on stationary, stable-spectrum data, where the extra branches only add capacity.

## Configure

- `enc_in`: number of channels.
- `patch_len`: must satisfy `patch_len <= seq_len + stride` (the series is repeat-padded by `stride`).

Other hyperparameters: preset defaults in `configs/models/DTAF.toml`; tune generically.

## Differences

Independent rewrite from the published TFS/FWM description; `decisionintelligence/DTAF` at `9d12aa40` (no license file, recorded `NOASSERTION`) was used as reference only, nothing copied.

- The expert MLPs are a portable local choice, and the forecast head is a direct flatten linear head; both are disclosed local choices rather than official modules.

Citation: Lu, J., Chen, P., Guo, C., Shu, Y., Wang, M., Yang, B. "Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing." AAAI 2026, pp. 24070-24078. doi:10.1609/aaai.v40i29.39585.
