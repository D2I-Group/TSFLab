---
name: "MultiPatchFormer"
description: "Transformer over four fused patch scales per channel, followed by a channel-wise encoder and a semi-autoregressive linear head. Use for multivariate data with patterns at several time granularities and correlated channels; not for short lookbacks, exogenous inputs, or probabilistic output."
---

# MultiPatchFormer

## Idea

- `PatchScale` embeds patches of lengths 8, 16, 24, 32 (strides 8, 8, 7, 6); outputs are interpolated to a common token count and concatenated, so one token stream sees several time granularities.
- Stacked `nn.TransformerEncoderLayer`s model temporal dependence per channel; the token mean then goes through one more encoder layer across channels.
- `SemiAutoregressiveHead` emits the horizon in up to 8 groups, each linear layer also seeing the groups emitted before it, to reduce over-fitting and noise.
- Series are standardized with per-window mean and standard deviation.

## When to use

- Multivariate series whose temporal correlations live at several scales, and whose channels interact (the channel encoder models inter-series relations).
- Lookbacks of at least 32 steps: shorter windows clip the larger patch scales to `seq_len`, collapsing the multi-scale view.
- Not for exogenous or calendar inputs (marks are ignored) or quantile output (point forecast only); channel attention is quadratic in the channel count.

## Configure

- `enc_in`: must equal the dataset's channel count. Patch scales are fixed; `seq_len` should be at least 32 so every scale is distinct.

Other hyperparameters: preset defaults in `configs/models/MultiPatchFormer.toml`; tune generically.

## Differences

- Clean-room implementation of the paper's structure: 8/16/24/32-point patch scales, aligned patch tokens, temporal Transformer stack, channel Transformer, and progressive semi-autoregressive horizon groups.
- The Time-Series-Library link is provenance only; no source was copied.
- `d_model` must be divisible by 4 (one quarter per scale) and by `n_heads`.
