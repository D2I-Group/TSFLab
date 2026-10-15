---
name: "HN_MVTS"
description: "Hypernetwork maps a learnable per-channel embedding to that channel's final-layer weights on a shared channel-independent temporal MLP. Use for multivariate forecasting where channel-specific heads help but full channel mixing overfits; not for modelling explicit cross-channel interactions."
---

# HN_MVTS

## Idea

- `channel_embedding` (one learnable vector per channel) goes through a one-hidden-layer `hypernetwork` that outputs the channel's `pred_len x d_model` projection weights and bias (`generated_projection`, Eqs. 2-4).
- A shared channel-independent `Linear(seq_len, d_model)` plus GELU is the base temporal encoder.
- The forecast is the per-channel einsum of encoder features with the generated weights, inside `revin`.
- Generating only the last layer restricts new parameters and acts as a data-adaptive regularizer across channels.

## When to use

- Designed for multivariate data where channel-dependent models degrade relative to channel-independent ones: the base stays channel-independent while the hypernetwork ties per-channel heads through a shared generator.
- The paper reports gains in long-range accuracy over the base models (DLinear, PatchTST, TSMixer).
- Small base model, suited to light compute budgets.
- Not for tasks that need explicit cross-channel interactions (no mixing in the encoder); point output only.

## Configure

- `enc_in`: the dataset's channel count (one embedding and one generated projection per channel).

Other hyperparameters: preset defaults in `configs/models/HN_MVTS.toml`; tune generically.

## Differences

- Follows Eqs. (2)-(4), checked against `src/layers.py` and `src/models/dlinear.py` of the official repository at the pinned revision; written for TSFLab, nothing copied.
- The base model is a compact channel-independent temporal MLP; the paper's alternative PatchTST/TSMixer backbones are not included.
- Embeddings are learned from random initialization rather than initialized with training-split Pearson/PCA statistics.
- The hypernetwork also runs at inference; the paper's training-only prior (no extra inference cost) is not separated.
