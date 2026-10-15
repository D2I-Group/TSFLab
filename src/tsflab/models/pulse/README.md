---
name: "PULSE"
description: "Phase-anchored MLP: subtracts a learned per-phase anchor, normalizes only the residual, and evolves the future anchor with a cross-attention phase router. Use for periodic series with distribution shift between training and inference; not for aperiodic data or when no stable period exists."
---

# PULSE

## Idea

- `phase_codebook` is a learned `[phase_period, channels]` table; `disentangle` subtracts the phase anchor and normalizes only the residual, then adds the anchor back.
- A small MLP `backbone` maps the normalized window to a latent future over time.
- `PhaseRouter` evolves the future anchor in two cross-attention stages (history queries future, then future queries the result) at `phase_resolution` and interpolates to the horizon; it mixes channels.
- The forecast is the de-normalized residual plus the future anchor, so statistics are restored in the residual coordinate only.
- `statistic_aware_mixup` and `frequency_mae` are provided as training utilities but are not called by `forward` or the default trainer.

## When to use

- Non-stationary periodic data: the paper decomposes each series into a deterministic phase structure and stochastic residual fluctuations, targeting distribution shifts between training and inference.
- Requires a stable period: the anchor table is indexed by position modulo `phase_period`.
- Not for aperiodic series; timestamps are not used (phase comes from position only).

## Configure

- `enc_in` follows the channel count: must equal the number of input channels (codebook columns and router width).
- `phase_period` follows the dataset period: the cycle length in steps that indexes the phase codebook.

Other hyperparameters: preset defaults in `configs/models/PULSE.toml`; tune generically.

## Differences

Compact randomly initialized realization written for TSFLab after inspecting `models/PULSE.py` at the pinned revision (no license file; nothing copied); not the released trained system.

- Statistic-Aware Mixup and Frequency-MAE exist as explicit training utilities but the generic forward path does not invoke them.
- Timestamp covariate encoding is omitted; phase positions are direct circular positions.
- Phase Router tokens are resampled to a fixed resolution.

Cite: Liu, Shao, Chen, Chen, Wang, Wu, "PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting", ICML 2026 (arXiv:2605.16793).
