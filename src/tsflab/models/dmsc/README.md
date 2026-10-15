---
name: "DMSC"
description: "Multi-scale forecaster: a data-dependent coarse-to-fine patch cascade with triad convolution fusion (intra-patch, inter-patch, cross-variable) and a global plus top-k local expert head fused across scales. Use for multivariate data with patterns at several time scales; not for short lookbacks or probabilistic output."
---

# DMSC

## Idea

- A small scale network picks a base patch length per batch between `patch_min` and `patch_max`; each cascade layer shrinks it by `patch_decay` (Eqs. 6-10).
- `TriadInteractionBlock` fuses an intra-patch conv, a dilated inter-patch conv and a cross-variable gate by softmax (Eqs. 11-16).
- Each layer sees the input plus a gated projection of the previous scale's feature (Eqs. 1-5).
- A mixture of always-on global experts and top-k local experts forecasts each scale; scale forecasts are fused with learned weights (Eqs. 17-21); a routing-entropy balance loss is added in training (Eqs. 22-23).

## When to use

- Multivariate series with structure at several time scales, with some cross-variable interaction.
- Needs lookbacks at least as long as `patch_max`; point output only.

## Configure

- `enc_in`: the dataset's channel count.
- `patch_max`: `patch_min <= patch_max <= seq_len`.

Other hyperparameters: preset defaults in `configs/models/DMSC.toml`; tune generically.

## Differences

- Independent rewrite after reading the pinned official code (no license); nothing copied.
- The many paper/code conflicts (balance-loss sign, history vector, router form, dead patch gradients, extra MLP residual, script defaults vs Table 2) are resolved and recorded as `issues` in `card.toml`.
- The `--use_res` per-scale expert residual is not implemented; the catalog trainer replaces the paper's training setup. Detail in reference.md.
