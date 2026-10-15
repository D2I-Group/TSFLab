---
name: "MoU"
description: "Noisy top-k mixture of linear patch extractors, then a Mamba, FFN, conv, attention hierarchy over channel-independent patch tokens. Use for long-term multivariate forecasting with long lookbacks and heterogeneous local patterns; not for cross-channel interaction, calendar inputs, or probabilistic output."
---

# MoU

## Idea

- `MixtureOfFeatureExtractors` (MoF) gives each patch a noisy top-k mixture of linear `patch_len -> d_model` extractors, so patches with different local patterns use different encoders at near-constant active cost.
- `MixtureOfArchitecturesBlock` (MoA) stacks Mamba (`mamba` component), a GELU feed-forward residual, a kernel-3 convolution residual, and one post-norm self-attention layer, widening the receptive field layer by layer.
- Tokens are channel-independent: RevIN, end-replication padding, unfolding into patches, a learned position table, then a flatten linear head (`flatten_forecast_head`).

## When to use

- Long-term forecasting with long lookbacks, where patch tokens keep the sequence short (official ETTh1 script: `seq_len = 336`).
- Series whose local patch patterns vary, so a per-patch mixture of extractors helps over a single patch embedding.
- Multivariate data with weakly related channels (channel-independent, shared weights).
- Not when cross-channel interaction carries the signal, when calendar marks matter (ignored), or when quantiles are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `patch_len`, `stride`: follow `seq_len`; `patch_len <= seq_len` is enforced, and with end padding the token count is `(seq_len - patch_len) // stride + 2`, which sets attention cost and head width.

Other hyperparameters: preset defaults in `configs/models/MoU.toml`; tune generically.

## Differences

- Independent rewrite of the `entype=mof`, `ltencoder=mfca` configuration from the paper and official code at `5a73fd7` (no license; nothing copied).
- MoA blocks are stacked sequentially (the official loop only uses the last block; see issues).
- Mamba is the portable pure-PyTorch `mamba` component, not the `mamba_ssm` CUDA kernel.
- The MoF gate computes all extractors densely and keeps top-k weights; no load-balancing loss (the official one is commented out), so training is MSE only.
- Attention is `nn.MultiheadAttention` in a post-norm (BatchNorm) layer rather than PatchTST's residual-attention layer.
- Series decomposition and the other `entype` / long-term encoder variants are not implemented. Details in reference.md.
