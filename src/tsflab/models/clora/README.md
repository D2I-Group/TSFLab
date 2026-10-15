---
name: "CLoRA"
description: "Per-channel low-rank adapters add an identity-aware code to each variate token before a shared channel-mixing backbone (iTransformer or RMLP). Use for multivariate data whose channels behave differently but interact; not for variable channel sets, transfer without fixed channel order, or probabilistic output."
---

# CLoRA

## Idea

- `ChannelLowRankAdapter` stores one rank-`r` factor per channel and a shared lift `W: r -> d`, giving a cheap per-channel adapter (Eq. 5).
- Each channel's shared token (inverted embedding, width `d_model - adaptation_dim`) is projected by its own adapter to a `d`-dim identity-aware code (Eq. 6).
- Shared embedding and code are concatenated into `[B, C, d_model]` tokens (Eq. 7) and mixed by a channel-dependent backbone; the adapter adds only `C*D*r + r*d` parameters.
- `backbone="itransformer"` (attention across variates, non-affine `revin`) or `"rmlp"` (residual MLP, affine `revin`); both end in a linear `d_model -> pred_len` projection.

## When to use

- Multivariate data where channels interact but differ in behaviour, so a shared backbone benefits from channel identity at small parameter cost.
- Channel set and order are fixed by `enc_in`; not for datasets with changing channels.
- Marks are ignored; point forecasts only.

## Configure

- `enc_in`: must equal the input channel count; one adapter per channel.

Other hyperparameters: preset defaults in `configs/models/CLoRA.toml`; tune generically (`rank`, `adaptation_dim` follow the official scripts).

## Differences

- Independent implementation; no official code copied.
- Only the `itransformer` and `rmlp` backbones of the paper's seven are provided.
- `revin` detaches the standard deviation (forward values and gradients identical); adapter init is Xavier-uniform as in the official code (the paper does not specify it).
- Adapter-only fine-tuning on a new dataset (Sec. 4.4) is a training procedure, not provided by the runtime.
