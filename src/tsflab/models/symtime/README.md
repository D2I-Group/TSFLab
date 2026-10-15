---
name: "SymTime"
description: "Downstream path of the SymTime foundation model: RevIN, moving-average trend/periodic split, patch Transformer on the periodic part plus a linear trend head. Use for channel-independent long-term forecasting trained from scratch; not for zero-shot use (no pre-trained weights) or cross-channel structure."
---

# SymTime

## Idea

- `series_decomposition` (moving average, kernel 25) separates a trend from the periodic remainder after `revin` normalization.
- The periodic part is cut into non-overlapping patches, embedded, and encoded by a pre-norm Transformer (`series_encoder`) with channels folded into the batch; `periodic_head` flattens patch tokens to the horizon.
- The trend goes through a direct linear `trend_head`; the two forecasts are added before denormalization.
- Only the downstream forecasting path exists: the symbol encoder and synthetic series-symbol pre-training of the paper are not in the runtime.

## When to use

- Long-term forecasting of series with a smooth trend plus a periodic component, channels treated independently.
- When a PatchTST-like baseline with an explicit trend branch is wanted; train from scratch on the target data.
- Not when the benefit of the paper's pre-training is the point (no S² pre-training or released weights), or when cross-channel interactions matter.

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.
- `patch_len`: non-overlapping patch length; ideally divides `seq_len` (otherwise the lookback is replicate-padded to `ceil(seq_len / patch_len)` patches).

Other hyperparameters: preset defaults in `configs/models/SymTime.toml`; tune generically.

## Differences

- The forecasting path was checked against `models/pretrain_model.py` and `models/finetune_model.py` at the pinned revision; the local module was written for TSFLab and copies no external source.
- Not a replacement for the 50B-scale S² pre-training: no symbol/DistilBERT encoder, momentum encoders, MLM/MTM/contrastive objectives, or pre-trained weights.
- The compact Transformer defaults to two rather than six layers.
