---
name: "APTF"
description: "PatchTST trained with loss-ranked bucket weights, staged bucket groups, and a co-trained amortization model. Use for training sets with hard-to-predict, low-predictability samples that dominate the loss; not for cheap training (two models are trained) or cross-channel modelling."
---

# APTF

## Idea

- Predictability-aware loss (Algorithm 1): samples ranked by per-sample squared error are split into `K` buckets and weighted from 1 downward, so the least predictable samples get the smallest weight.
- Hierarchical loss (Sec. 3.1.4): at stage `S` it averages `S+1` bucket groups, group `i` using `K-i` buckets, normalized by `G = K` stages.
- Stages advance every `stage_epochs` epochs (Eq. 2); the objective counts training steps to know the epoch.
- Amortization (Algorithm 2): an independently initialised model of the same architecture is co-trained; each model's loss is bucketed by the other's errors to remove predictability-estimation bias.
- The forecaster is the shared `patchtst` backbone; `forward` runs only the source model.

## When to use

- Training data contains low-predictability samples (noise, anomalies, abrupt events) whose large errors would otherwise dominate gradients.
- A training-time framework: it changes the loss, not the architecture, and roughly doubles training compute.
- Channel-independent PatchTST backbone; point output only.

## Configure

- `enc_in`: number of channels.
- `patch_len` (with `stride`): backbone patch length, at most `seq_len`.

Other hyperparameters: preset defaults in `configs/models/APTF.toml`; tune generically. `d_model` must be divisible by `n_heads`.

## Differences

Independent rewrite of Sec. 3 and Appendix A.1.1 after reading the pinned official code (no license file).

- The paper's `K = 9` first-stage buckets and `G = K` stages are used (the official code builds `K - 1`).
- Ranking, remainder handling, empty buckets, and the fixed `1/G` normalizer follow the official code.
- Bucket losses use the run's configured criterion; ranking always uses per-sample MSE.
- Both models share the catalog optimizer instead of one AdamW each, so adaptive optimizer state and clipping act jointly.
- Backbone fixed to `patchtst` (one of the paper's eleven); classification is not included.

Full detail in `reference.md`.
