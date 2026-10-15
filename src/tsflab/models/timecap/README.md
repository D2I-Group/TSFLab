---
name: "TimeCAP"
description: "Channel-aware patch Transformer: overlapping channel groups attend within time-aligned patches and talk through meta-router tokens; GRU and one-shot heads are blended over the horizon. Use for multivariate data with grouped cross-channel dependencies; not for weakly correlated or very many channels."
---

# TimeCAP

## Idea

- Channels are cut into overlapping groups (`group_size`, `group_stride`, wrapping around); each group has its own patch projection and a learned meta-router token per patch.
- `channel_aware_mask` restricts intra-group attention to tokens of the same patch, and router-to-group attention to the same patch, so communication across groups goes through router tokens only.
- Group outputs are scatter-averaged back to channels.
- `dual_head_forecast` blends a GRU autoregressive head and a flatten one-shot linear head with a sigmoid weight that shifts toward one-shot over the horizon (`fusion_alpha`, `fusion_midpoint`).

## When to use

- Multivariate forecasting where dependencies among variables matter, with local dependencies inside groups of related channels and global coherence across groups.
- Horizons where early steps benefit from autoregressive decoding and later steps from one-shot prediction.
- Not for weakly correlated channels (the channel mixing adds little), very many channels (one projection per group), or zero-/few-shot use: no pre-trained checkpoint is included.

## Configure

- `enc_in`: number of channels; groups start every `group_stride` channels, giving `ceil(enc_in / group_stride)` groups.
- `group_size`: channels per group, clamped to `enc_in`; with `group_stride < group_size` groups overlap.
- `patch_len`: ideally divides `seq_len` (otherwise the lookback is replicate-padded at the end).

Other hyperparameters: preset defaults in `configs/models/TimeCAP.toml`; tune generically.

## Differences

- Checked against the pinned `models/TimeCAP.py` and `layers/TimeCAP_EncDec.py`; the local module was written for TSFLab and copies no external source.
- A compact, randomly initialized forecasting rewrite, not the multi-domain pre-trained checkpoint.
- One channel-aware routing stage and a GRUCell autoregressive head; no pre-training/fine-tuning loss schedule or self-distillation.
