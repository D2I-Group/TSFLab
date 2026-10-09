---
name: "TimeMosaic"
description: "Patch Transformer for temporal heterogeneity: each lookback region softly chooses its patch size by information density, and each horizon segment gets its own prompt and linear head. Use for series mixing dense and stable stretches, with horizons of differing difficulty; not for cross-channel structure."
---

# TimeMosaic

## Idea

- `adaptive_patch_tokens` splits the history into regions of the largest patch size; a `granularity_classifier` softmax chooses among `patch_sizes` per region, and candidate embeddings are repeat-aligned to a common token count and blended.
- Each horizon segment has a learned prompt token prepended to the keys and values of a shared attention layer (`segment_prompts`).
- Each segment gets its own linear head over the flattened tokens (`segment_heads`); segments are concatenated. Channels share weights; `revin` wraps the model.

## When to use

- Series whose local dynamics are heterogeneous: information-dense stretches benefit from fine patches, stable ones from coarse patches.
- Horizons where near and far steps differ in difficulty (segment-wise decoding treats each horizon segment as a related subtask).
- Channel-independent and trained from scratch: not for cross-channel or covariate-driven targets, and not a substitute for the paper's large-corpus pre-trained model.

## Configure

- `enc_in`: number of channels.
- `patch_sizes`: each size must divide the largest; the history is cut into regions of the largest size, and a
  `seq_len` that is not a multiple of it is left-padded with the first value.
- `num_segments`: horizon segments, clamped to `pred_len`; segment lengths differ by at most one step.

Other hyperparameters: preset defaults in `configs/models/TimeMosaic.toml`; tune generically.
The preset follows the official `run.py` defaults: `d_model` 512, 8 heads, patch sizes 8/16/32.

## Differences

- Checked against the pinned `models/TimeMosaic.py` and `models/Mosaic.py`; the local module was written for TSFLab and copies no external source (repository has no license file, `NOASSERTION`).
- The adaptive patch equations map to a soft region-wise mixture over configured granularities, not Gumbel-hard selection; no budget regularization.
- Segment prompts are applied only to attention keys and values before separate forecast heads.
- End-to-end local training instead of a frozen foundation backbone; the 321-billion-observation pre-training corpus is not used.
- Lookbacks that are not a multiple of the largest patch size are left-padded with the first value; the paper uses multiples.
