---
name: "SRSNet"
description: "Channel-independent patch model that scores, gates, and soft-reorders patches into a selective representation space before a linear head under RevIN. Use for patch-based forecasting where only some parts of the lookback are informative; not for cross-channel dependencies or probabilistic output."
---

# SRSNet

## Idea

- Conventional patching uses fixed adjacent patches; a selective representation space adaptively selects and shuffles the most informative patches of the context.
- `SelectivePatching` embeds each patch and scores it with a small MLP, gating embeddings with a sigmoid of the mean-centered score (`alpha`).
- `DynamicReassembly` turns scores into a soft rank and soft permutation (`assignment`) that reorders patches differentiably.
- `SelectiveRepresentationSpace` adds optional positions and LayerNorm (no attention); `flatten_forecast_head` maps the flattened patches to the horizon inside `revin`.

## When to use

- Lookbacks where informative segments are unevenly placed, so weighting and reordering patches helps a simple head.
- Lightweight alternative to patch Transformers (MLP scoring, no attention).
- Channel-independent with shared weights: not for data whose signal lies in channel interactions. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly (RevIN and head width).
- `patch_len` and `stride` follow `seq_len`: `patch_len <= seq_len`; patch count is `floor((seq_len - patch_len) / stride) + 1`.
- Other hyperparameters: preset defaults in `configs/models/SRSNet.toml`; tune generically.

## Differences

Clean-room rewrite; the official `decisionintelligence/SRSNet@6ee35d4` (MIT) is reference-only and was not copied. Mapping: Selective Patching -> `SelectivePatching`, Dynamic Reassembly -> `DynamicReassembly`, SRS -> `SelectiveRepresentationSpace`, MLP head -> shared `FlattenForecastHead`.

- Dynamic Reassembly uses a soft-sort relaxation.
- Forecast-only interface; only the paper's linear head (`head_mode = "linear"`) is supported.

Citation: Wu, Qiu, Cheng, Li, Hu, Guo, Yang, "Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective", NeurIPS 2025, arXiv:2510.14510.
