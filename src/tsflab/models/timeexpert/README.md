---
name: "TimeExpert"
description: "Channel-independent patch Transformer whose attention is a temporal mixture of experts: each query patch attends only to its top-k most relevant patch positions, optionally plus a shared global expert. Use for long-term forecasting where only a few past segments are relevant; not for cross-channel structure."
---

# TimeExpert

## Idea

- `TopKExpertAttention` treats each key/value patch position as an expert; each query routes to its top-k most relevant ones, with an optional shared global expert (`shared`).
- `TMOEBlock` pairs it with a residual feed-forward network, stacked `e_layers` times over `PatchEmbedding` tokens (end-padded by one stride).
- Each channel is standardized over the lookback and folded into the batch, so all channels share weights; `FlattenForecastHead` maps patch tokens to the horizon.

## When to use

- Long lookbacks where each future segment depends on a few relevant past segments rather than on all of them; sparse top-k routing filters irrelevant patches.
- Channels that can be modelled independently with shared weights.
- Not when cross-channel interactions, covariates, or calendar effects drive the target.

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.
- `patch_len`, `stride`: give `(seq_len - patch_len) // stride + 2` patches.
- `topk`: at most the patch count above (each query keeps `topk` key positions).

Other hyperparameters: preset defaults in `configs/models/TimeExpert.toml`; tune generically.

## Differences

- Independent rewrite; the official repository has no LICENSE file (`NOASSERTION`) and was consulted only for paper details (see THIRD_PARTY_NOTICES.md). The TMOE routing/gathering/attention data flow was re-derived from the pinned official file's class boundaries and docstrings; no source lines copied.
- The top-k expert attention is the cataloged, paper-neutral `topk_expert_attention` component, since scoring every position, keeping the top-k per query and optionally appending a shared expert does not depend on the patch embedding or head.
- The official code offers `mul_weight` modes "soft" and "none"; this implementation matches `mul_weight='none'` composed with softmax attention (router weights scale the gathered keys before the second softmax and are not reapplied).
- Only forecasting is implemented (the official model also covers imputation, anomaly detection, classification).
- The preset lowers `d_model` from the official 512 to 128 to match the catalog's other patch Transformers; `n_heads=8`, `patch_len=16`, `stride=8`, `topk=4`, `shared=False` are unchanged.
