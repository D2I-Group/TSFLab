---
name: "MixLinear"
description: "Ultra-lightweight channel-shared linear forecaster mixing a segment-domain time path and a low-rank spectral path on a downsampled series. Use for long-term forecasting under tight compute or memory budgets (edge devices); not when cross-channel interactions or calendar effects drive the target."
---

# MixLinear

## Idea

- Centers each series and average-pools it by `downsample`, shrinking the length before the linear layers.
- `SegmentTrendPath` splits the reduced series into segments, encodes each to a small `hidden_rank`, mixes across segments linearly, and decodes (intra- and inter-segment variation).
- `LowRankSpectralPath` applies a complex rank-constrained operator `U(VF)` to the FFT of the reduced series (`spectral_rank`), cutting core parameters from O(n^2) to O(n).
- The two path outputs are summed and linearly interpolated to `pred_len`; weights are shared across channels.

## When to use

- Designed for long-term forecasting on resource-constrained devices: tiny parameter count and compute.
- Very few parameters also limit overfitting when training windows are scarce.
- Channels are processed independently with shared weights and no calendar input, so it cannot exploit cross-channel or timestamp structure.
- Downsampling discards fine-grained, high-frequency detail.

## Configure

- `enc_in`: number of channels (weights are shared).
- `downsample`: `seq_len` must be divisible by it.
- `segments`: `seq_len / downsample` must be divisible by it; `spectral_rank` must not exceed `seq_len / downsample`.

Other hyperparameters: preset defaults in `configs/models/MixLinear.toml`; tune generically.

## Differences

- Implemented from paper Eqs. (1)-(5); the unlicensed repository (`models/MixLinear.py`) was inspected at the pinned revision, no source copied.
- Unspecified adaptive downsampling and reconstruction are replaced by fixed average downsampling, a symmetric local encoder/decoder, linear interpolation to `pred_len`, and per-series centering.
- The paper's exact 0.1K parameter count and benchmark numbers are not claimed.
