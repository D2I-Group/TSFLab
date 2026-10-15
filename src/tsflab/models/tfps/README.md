---
name: "TFPS"
description: "Patch model for patch-level distribution shift: time and Fourier patch encoders, subspace clustering of cross-variate patch features, and pattern-specific MLP experts per cluster. Use for long-term forecasting where local patterns change within and across series; not for very short horizons or many channels."
---

# TFPS

## Idea

- `patchify` end-pads each RevIN-normalized variate by one stride and cuts `N = floor((L - P) / S) + 2` patches (Section 3.3); each branch embeds them linearly with a learnable position table.
- Dual-Domain Encoder (Section 3.4): `TimeEncoderLayer` is a post-norm BatchNorm Transformer block with residual attention scores (Eq. 1); `FrequencyEncoderLayer` replaces attention with the real part of a 2-D FFT over hidden and patch axes (Eq. 2).
- `PatternIdentifier` clusters cross-variate patch features `z_i in R^{C D}` with `K` subspace bases (Eqs. 3-9); `PatternExperts` (MoPE) gates two-layer MLP experts per patch with `Softmax(TopK(s))` (Eqs. 10-11).
- The frequency branch returns to time by an inverse 2-D FFT; both branches are concatenated and mapped by per-variate flatten-linear heads.
- `training_objective` adds both branches' clustering losses to the forecasting criterion (Eq. 12); validation and test use the forecast only.

## When to use

- Series whose local patterns shift between patches (patch-level distribution shift, regime-like segments); experts specialise per identified pattern.
- Multivariate data of moderate width: clustering features are `enc_in * d_model` wide, and experts are `q -> 4q -> q` MLPs, so cost grows quickly with channels.
- Horizons long enough to hold at least one output patch.
- Not for probabilistic output or for very many channels.

## Configure

- `enc_in`: number of channels; `enc_in * d_model` must be at least the number of experts.
- `patch_len`, `stride`: `patch_len <= seq_len`; `pred_len` must give at least one output patch (`floor((pred_len - patch_len) / stride) + 2 >= 1`). Input and output patch counts follow `seq_len` and `pred_len`.

Other hyperparameters: preset defaults in `configs/models/TFPS.toml`; tune generically.

## Differences

- Independent rewrite from Section 3 (Eqs. 1-12) and Appendix L (Algorithm 1) after reading the pinned official code (no license file, `NOASSERTION`); nothing copied or imported.
- Each clustered row is one output patch's `C x D` feature (the official reshape mixes patches and variates).
- Subspace bases are standard Gaussian, shape `q x K floor(q / K)` (official: uninitialised square tensor requiring `K | q`).
- Deterministic gating per Eq. (10) (official router adds noise in training and evaluation).
- Clustering-loss scales follow the official code (`eta = 5`, unsquared norms scaled `1e-3`, mean KL with `beta = 0.1`), not the paper's.
- Full detail: reference.md, `## Differences in detail`.
