---
name: "CATS"
description: "Cross-attention-only Transformer: learned future-patch queries attend embedded history patches, with heavy parameter sharing and query-adaptive masking. Use for long-horizon multivariate forecasting under parameter or memory budgets; not for exploiting cross-channel dependence or probabilistic output."
---

# CATS

## Idea

- No self-attention: `CrossAttentionLayer` has learned `future_queries` (one per output patch) attend the embedded history patches.
- The same embedding, attention and projection weights serve every horizon patch and every channel (`query_independence=False` shares queries across channels).
- Query-adaptive masking randomly drops attention outputs in training with probability rising linearly from `QAM_start` to `QAM_end` across layers.
- The series is centered on its last value (not detached) and the level is added back after the linear patch projection.

## When to use

- Long-horizon forecasting where a lean model is wanted: parameters are shared across horizon patches and channels, reducing parameters and memory versus self-attention Transformers.
- Channels are processed independently; it does not model cross-channel correlation.
- Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count; it sizes per-channel queries when `query_independence = true`.

Other hyperparameters: preset defaults in `configs/models/CATS.toml`; tune generically.

## Differences

- Paper-driven local implementation; the MIT repository is reference-only and no source file was copied or adapted.
- History patches give keys and values; learned future-patch parameters give the only queries; every layer is cross-attention only.
- Embedding, attention and output parameters are shared across horizons; the query-adaptive stochastic mask applies to the attention residual in training.
- Last-value centering is not detached, so it stays model-local rather than using `last_value_center` (see reference.md).
