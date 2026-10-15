---
name: "CoRA"
description: "Correlation-aware adapter on a small linear base: a batch Pearson prior plus polynomial low-rank dynamic correlation, split into positive and negative parts, gates a residual correction. Use for multivariate data with meaningful channel correlations; not for univariate series or reproducing the TSFM fine-tuning setup."
---

# CoRA

## Idea

- `dynamic_correlation` adds a time-varying low-rank term (polynomial coefficients from each channel's encoding, shared time-invariant composition matrices) to the batch Pearson correlation.
- Positive and negative correlations are separated; each side is softmax-normalized and aggregates values projected by `_HeterogeneousProjection`.
- The fused context gives a gated correction added to a `base` linear forecast.

## When to use

- Multivariate data where channels are correlated, positively or negatively, and a channel-independent base forecaster misses that.
- Lightweight: the adapter adds a low-rank correction, not full channel attention.
- Point forecasts only; univariate data gains nothing.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/CoRA.toml`; tune generically.

## Differences

- Clean-room runtime of the polynomial time-varying factor, low-rank time-invariant composition, Pearson prior, positive/negative projections, fusion and gated residual forecast.
- A small local linear forecaster substitutes for the paper's pre-trained TSFM.
- The training-only H-PCorr dual contrastive objective and the TSFM fine-tuning protocol are not implemented.
- The official `ts_benchmark/plugin/plugin.py` (no license) was inspected at the pinned revision; nothing was copied.
