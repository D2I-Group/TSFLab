---
name: "CANet"
description: "Multi-patch-size branches with adaptive spectral filtering, non-stationary style restoration, and dual-path conv. Use for long-term forecasting of non-stationary series whose level and scale drift; not for cross-channel or probabilistic tasks."
---

# CANet

## Idea

- One `CANetLayer` branch per patch size (default 8 and 64, half-stride overlapping patches), concatenated into one head.
- `AdaptiveSpectralBlock` filters patch embeddings with learned complex weights and an energy-threshold high-pass mask (straight-through estimator).
- NSAN (`adain_style_norm`) re-injects per-sample mean/std "style", blended from the window's own statistics and a parallel style embedding (`StyleBlendingGate`), instead of a fixed affine; instance normalization therefore does not discard the distribution shift (paper Sec. 3).
- `InteractiveConvolutionalBlock` mixes patches with kernel-1 and kernel-3 convolution paths fused by cross-multiplication.

## When to use

- Long-term forecasting under non-stationarity, where instance normalization alone removes useful level and scale information.
- Several temporal scales (short and long patches) and frequency-selective filtering of noise.
- Channel-independent with shared weights; point output only.

## Configure

- `enc_in`: number of channels.
- `patch_sizes`: each at most `seq_len`.

Other hyperparameters: preset defaults in `configs/models/CANet.toml`; tune generically.

## Differences

The official `canet/canet.py` and `canet/modules/` (MIT) were inspected at the pinned revision to resolve the patch-style-spectral-NSAN-ICB execution order, the straight-through frequency mask, and the NSAN statistic-blending equations, which the paper leaves underspecified; nothing copied.

- Only the paper's default normalization pairing (`LayerNorm` before the spectral block, NSAN after) is implemented; the official BatchNorm/InstanceNorm/DAIN ablation options are not.
- The official `StackedKroneckerProductLayer` (Kronecker-factored branch-concatenation projection) is a dense `nn.Linear`: more parameters, same computation.
- Stochastic depth (`DropPath`) on the branch residual is omitted (identity).
