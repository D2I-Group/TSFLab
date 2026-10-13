---
name: "TiDE"
description: "Time-series Dense Encoder: residual MLP encoder-decoder over the normalized lookback plus projected past/future calendar features, a per-step temporal decoder, and a global linear skip. Use for fast long-term forecasting where calendar covariates matter; not for cross-channel structure or probabilistic output."
---

# TiDE

## Idea

- Past and future time features are projected per step (`feature_projection`), flattened with the normalized lookback and fed to a stack of `ResidualBlock`s (encoder) and a dense decoder that emits one vector per horizon step.
- A per-step `temporal_decoder` combines each decoded vector with that step's future covariate projection.
- A global `Linear(seq_len, pred_len)` residual on the lookback is added; every channel runs through the same weights, and per-channel lookback statistics are undone at the end.

## When to use

- Long-term forecasting where linear-model speed is wanted but covariates and nonlinear dependencies should still be used; the paper reports 5-10x faster than the best Transformer at matching accuracy.
- Data with calendar effects: known future time features enter the encoder and the per-step decoder.
- Channel-independent: not for data where cross-channel interactions drive the target; static attributes are not supported.

## Configure

- `time_feat_dim`: number of mark features per step; must match the marks the runner supplies (6 raw calendar stamps by default). Missing marks are replaced by zeros.

Other hyperparameters: preset defaults in `configs/models/TiDE.toml`; tune generically.

## Differences

- Independent clean-room implementation from the paper; the THUML repository is reference-only and no source was copied.
- The scalar temporal decoder omits LayerNorm: in the THUML code, LayerNorm over one value removes it, so the nonlinear branch never reaches the forecast.
- `decoder_output_dim` is an internal width.
- Shared channel weights execute in bounded channel batches. Large training panels use activation checkpointing; outputs and gradients match the original per-channel equations with dropout disabled. Batching changes the seeded dropout draw order, while checkpoint recomputation preserves each forward draw.
- Static attributes, paper preprocessing, and a numerical comparison with reported results are not included.
