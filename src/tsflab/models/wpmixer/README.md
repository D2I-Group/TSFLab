---
name: "WPMixer"
description: "Channel-independent MLP mixer over a multi-level wavelet split: each resolution is patched, token/feature-mixed, and forecast directly, then fused by softmax weights. Use for efficient long-term forecasting from long lookbacks with multi-scale structure; not for cross-channel dependence or probabilistic output."
---

# WPMixer

## Idea

- `OrthogonalWaveletAnalysis` computes a fixed multi-level Haar/db1/db2 decomposition (approximation plus details) with strided convolutions.
- One `ResolutionMixer` per resolution: patch embedding, a token-mixing MLP over patches, a feature-mixing MLP, and a flatten linear forecast to the full horizon.
- Resolution forecasts are combined with learnable softmax weights (`resolution_logits`) rather than an inverse wavelet transform, inside `revin`; `no_decomposition=True` leaves a single branch.

## When to use

- Designed for long-term forecasting: patching lets a long lookback be used cheaply, and the MLP mixers keep compute low.
- Wavelet resolutions separate coarse trends from fine detail in both time and frequency.
- Channels share weights and are modelled independently.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count.
- `level`: each wavelet level roughly halves the resolution length, so `seq_len / 2**level` should stay well above `patch_len`.
- `patch_len`, `stride`: applied to every resolution; patch count is `ceil((length - patch_len) / stride) + 1` with zero padding, and `patch_len` is capped at the resolution length.

Other hyperparameters: preset defaults in `configs/models/WPMixer.toml`; tune generically.

## Differences

- Clean-room rewrite from the paper; the official code was not copied.
- Fixed Haar/db1/db2 analysis filters are local; branch forecasts are fused directly instead of reconstructed with an external inverse-wavelet package.
- `ResolutionMixer` is post-norm with its own patch embedding, unlike the cataloged pre-norm `mixer_block`, so it stays model-local.
