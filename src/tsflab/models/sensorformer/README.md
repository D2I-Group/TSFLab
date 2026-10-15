---
name: "Sensorformer"
description: "PatchTST-style patch Transformer with two-stage sensor attention: per-variable summary tokens, then every patch attends back to them. Use for multivariate data with cross-variable dependence across many channels at sub-quadratic cost; not for non-stationary level shifts (no instance norm) or probabilistic output."
---

# Sensorformer

## Idea

- `PatchEmbedding` (`embed`) replicate-pads by `stride`, extracts overlapping patches per variable, and embeds them with positions.
- `GlobalPatchCompressionAttention` stage one uses each variable's last patch as the query over all variables' patches to build one Sensor summary token per variable.
- Stage two lets every patch of every variable attend to those summaries, capturing cross-variable and cross-time dependence without full quadratic cross-patch attention.
- `FlattenForecastHead` maps each variable's final patches to the horizon; there is no instance normalization.

## When to use

- Multivariate data where variables influence each other with time lags: patches attend across variables through per-variable summaries.
- Many channels: compression to one summary token per variable avoids full attention over all patches of all variables.
- No instance normalization, so train/test level or scale shifts are not compensated inside the model. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `patch_len` follows `seq_len`: `patch_len <= seq_len + stride` (after replicate padding by `stride`).
- Other hyperparameters: preset defaults in `configs/models/Sensorformer.toml`; tune generically.

## Differences

No official code: the repository linked by the paper (`BigYellowTiger/Sensorformer`) is empty (zero commits), so no revision can be pinned and no reference comparison applies.

- Implemented from Section 3, Fig. 2, and Algorithms 1-2, which specify the patching ("the same patch strategy as PatchTST"), the two-stage attention's queries, keys, and values, and the LayerNorm/MLP wrapping of each stage.
- Patch length, stride, `d_model`, heads, feed-forward width, layer count, and normalization before patching are not given in the paper; they are ordinary parameters with PatchTST-style defaults.
