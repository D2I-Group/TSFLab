---
name: "Crossformer"
description: "Transformer over a channels-by-segments token array with two-stage attention (across time, then across variables via routers) and hierarchical segment merging. Use for multivariate data with strong cross-variable dependence; not for weakly correlated channels or probabilistic output."
---

# Crossformer

## Idea

- `dsw_embed` cuts every variable into segments and projects each to a token, giving a channels-by-segments array with learned positions (Dimension-Segment-Wise embedding).
- `TwoStageAttention` attends across segments within each channel, then across variables through a few learned routers (sender then receiver), avoiding quadratic cross-variable cost.
- `SegmentMerge` merges neighbouring segments between layers to build a hierarchy.
- A linear head per level forecasts directly and the levels are averaged.

## When to use

- Multivariate forecasting where cross-variable dependence carries signal; router attention keeps cross-variable cost linear in channel count.
- On weakly correlated channels, channel-independent models are usually the safer choice.
- Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count.
- `seg_len`: at most `seq_len`; lookbacks not divisible by it are front-padded with the first value.

Other hyperparameters: preset defaults in `configs/models/Crossformer.toml`; tune generically (`d_model` divisible by `n_heads`).

## Differences

- Clean-room implementation with the Apache-2.0 code as reference only: DSW segmentation, TSA temporal stage, router send/receive stage, hierarchical segment merging.
- Scale-wise direct linear heads, averaged, replace the paper's hierarchical decoder (HED); this is a disclosed simplification.
