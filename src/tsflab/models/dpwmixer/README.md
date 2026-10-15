---
name: "DPWMixer"
description: "Haar wavelet pyramid (no pooling loss); each scale gets a linear-trend path plus a patch-MLP path, fused by softmax weights. Use for channel-independent long-term forecasting where structure spans several resolutions; not for tasks needing cross-channel modelling."
---

# DPWMixer

## Idea

- `wavelet` (`DecimatedWaveletTransform`, Haar) is applied once per level, keeping the approximation signal, to build a multi-resolution pyramid without average-pooling information loss.
- `DualPathTrendMixer` forecasts each resolution by combining a global linear trend path with a patch-embedding MLP path, mixed by two learnable scalars (`path_weight`).
- Per-scale, per-channel forecasts are fused by softmax over learned `fusion_weight` logits; `revin` wraps the model; channels share weights.

## When to use

- Designed for long-term forecasting where both a global trend and local patch patterns at several resolutions matter.
- Lightweight MLP model; a reasonable choice under tight compute.
- Channel-independent: does not model cross-channel dependence.

## Configure

- `enc_in`: number of channels (the fusion logits are learned per channel).
- `patch_len` / `stride` are clamped to each resolution's length, so they impose no divisibility constraint on `seq_len`.

Other hyperparameters: preset defaults in `configs/models/DPWMixer.toml`; tune generically.

## Differences

Clean-room re-derivation from the paper and the module boundaries and defaults of `hit636/DPWMixer` at `0a787be2` (Apache-2.0); no source copied.

- The official `HaarWaveletSplit` (a fixed one-level Haar analysis filter applied iteratively) is replaced by one call per level of the cataloged `DecimatedWaveletTransform` (its other wavelets and exact inverse are unused).
- The official patch stage keeps `patch_len`/`stride` at every resolution and zero-pads short inputs; here they are clamped to the resolution's length so very short scales never get a degenerate patch count.
- The official default `d_model` is 64; the preset keeps `patch_len = 16`, `stride = 8` and raises `d_model` to 128 to match the catalog's other mixer presets.
- The fusion softmax over `(down_sampling_layers + 1, enc_in)` logits matches the official `Softmax(dim=0)`.
