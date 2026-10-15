---
name: "iTransformer"
description: "Inverted Transformer: each variate's whole lookback is one token, attention runs across variates and the FFN per variate. Use for multivariate data with correlated channels and long lookbacks; not for weakly correlated channels or univariate series."
---

# iTransformer

## Idea

- `InvertedEmbedding` projects a variate's entire `seq_len` series to one `d_model` token (`nn.Linear(seq_len, d_model)`); the six calendar mark columns, if given, become extra auxiliary tokens.
- `InvertedEncoderLayer` is a post-norm `nn.MultiheadAttention` over the variate axis, so attention captures inter-channel correlation while the feed-forward network learns per-variate temporal features.
- A linear `projection` maps each variate token to `pred_len`; auxiliary tokens are dropped.
- Each variate is standardized over its window and restored on the output (`use_norm`).

## When to use

- Multivariate data whose channels are correlated: attention over variate tokens is how cross-channel information enters.
- Long lookbacks are cheap: the window is embedded linearly, so cost grows with the variate count, not the lookback length.
- Attention is quadratic in the number of variates, so very many channels raise cost.
- Little benefit on univariate series or weakly correlated channels, where the channel attention has nothing to model.

## Configure

- `enc_in`: number of data channels (variate tokens); fixed per dataset.

Other hyperparameters: preset defaults in `configs/models/iTransformer.toml`; tune generically.

## Differences

- Clean-room implementation; the MIT repository is reference-only and nothing was copied.
- Whole-series variate tokens map Eq. (1), token-feature LayerNorm Eq. (2), attention runs on the variate axis.
- Optional six-column marks become auxiliary tokens.
- Channel count is fixed; efficient-attention plugins, non-forecast tasks, and checkpoint or metric reference comparison are omitted.
