---
name: "FDNet"
description: "Local-only weight-normalized 1x1/3x1 conv features per value with focal input decomposition: the latest sub-sequence gets the deepest branch. Use for efficient, robust channel-independent forecasting from long lookbacks; not for tasks needing cross-variate or global correlation."
---

# FDNet

## Idea

- Focal input decomposition (Section 5.2, Fig. 6): `focal_parts` (f) consecutive sub-sequences with proportions 1/2^(f-1), 1/2^(f-1), 1/2^(f-2), ..., 1/2 from latest to farthest.
- Each branch (Section 5.1, Fig. 4): 1x1 embedding of each value to `d_model`, decomposed feature extraction layers, per-variate flatten of `[time, d_model]`, horizon projection shared by all variates.
- `DecomposedFeatureExtractor` (Fig. 5): four weight-normalized 2D convolutions (1x1, 3x1, 1x1, 3x1) with GELU in two residual pairs; the variate kernel is one, so no global or cross-variate correlation is extracted.
- Closer sub-sequences get deeper branches; the branch forecasts are averaged.

## When to use

- Designed for efficient and robust forecasting with long inputs (official scripts: 672 steps, f = 5): recent history is modelled in detail, distant history coarsely.
- Purely local, channel-independent convolutions; it cannot exploit cross-variate structure.
- No instance normalization: relies on the loader's global z-scoring, so level shifts between splits are not corrected.

## Configure

- `enc_in`: number of channels.
- `focal_parts`: `seq_len` must be divisible by `2^(focal_parts - 1)`; use more parts for longer lookbacks (official: 5 for 672 inputs, 1 for 96).
- `num_layers` must be at least `max(focal_parts - 1, 1)`.

Other hyperparameters: preset defaults in `configs/models/FDNet.toml`; tune generically.

## Differences

Independent rewrite of Sections 5.1-5.2 and Figs. 4-6 after reading `OrigamiSL/FDNet` at `b636bd09` (Apache-2.0 license file governs over the CC BY-NC-SA README badge); nothing copied.

- Depth schedule follows the code (two latest sub-sequences get N layers, then N-1, ...), not Fig. 6 (N, N-1, ..., N-f+1).
- The official unused extra branch is not built.
- The ICOM/pooling ablation (FUNet) and the calendar `timebed` input (unused by every script) are not implemented.
- Training uses the configured loss (official loop: MSE + MAE, Adam, learning rate 1e-4 halved every epoch, batch 16). Details in `reference.md`.
