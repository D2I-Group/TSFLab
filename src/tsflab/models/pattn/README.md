---
name: "PAttn"
description: "Minimal patch Transformer from the LLM-ablation study: one self-attention layer on patch tokens, no positional encoding or FFN, linear head. Use as a cheap attention baseline for LLM-based or patch forecasters; not for cross-channel dependence, exogenous inputs, or probabilistic output."
---

# PAttn

## Idea

- Overlapping patches (default length 16, stride 8, replicate-padded) are linearly embedded with `patch_projection`.
- A single `nn.MultiheadAttention` layer with residual LayerNorm processes the tokens, with no positional embedding and no feed-forward sublayer.
- Tokens are flattened into `forecast_projection`, a single linear layer to `pred_len`.
- Channels are folded into the batch, and each window is standardized with its own mean and standard deviation.

## When to use

- As the paper's control: a cheap patch-attention encoder that matches or beats LLM-based forecasters, so use it to check whether a heavier model earns its cost.
- Tight compute budgets: one attention layer, no pretrained weights.
- Not when cross-channel dependence carries the signal (channel-independent), when timestamps or covariates matter (marks are not used), or when quantiles are needed (point output).

## Configure

- `patch_len`, `stride`: follow `seq_len`; `patch_len` must not exceed `seq_len`, and the token count is `(seq_len + stride - patch_len) // stride + 1` (end padded by `stride`).

Other hyperparameters: preset defaults in `configs/models/PAttn.toml`; tune generically.

## Differences

- Independent clean-room implementation from the paper's Figure 4 and appendix D.3; the pinned repositories are reference-only and no source was copied.
