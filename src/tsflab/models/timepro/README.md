---
name: "TimePro"
description: "Variate-axis selective state-space model: each channel's patch embeddings form one token, a bidirectional scalar-state scan runs across variates, and its state is mixed over a (time-patch, variate) grid. Use for efficient multivariate long-term forecasting with cross-variable dependencies; not for univariate data."
---

# TimePro

## Idea

- Each channel is patched and embedded; all patch embeddings of a channel are concatenated into one token per variate, so the scan runs over the variate axis.
- `ProMamba` runs a diagonal selective scan (`diagonal_selective_scan`, scalar state, `d_state=1`) and reshapes the hidden state into a (time-patch, variate) grid that `GridStateMixer` mixes locally before read-out: the variable- and time-aware hyper-state.
- `ProBlock` runs forward and reversed scans plus an MLP; a LayerNorm and linear projector emit the horizon. Inputs are instance-standardized (`use_norm`).

## When to use

- Multivariate long-term forecasting where dependencies across variables matter and their importance varies over time patches; the scan is linear in the number of variates.
- Efficiency-sensitive settings: the paper targets low cost; note the local scan is a portable sequential recurrence (slower than the CUDA kernel).
- Not for univariate data (the scan axis is the variates), covariates, or probabilistic output.

## Configure

- `enc_in`: number of channels = scan length.
- `patch_len`, `stride`: `patch_len <= seq_len`; the end-padded patch count `(seq_len + stride - patch_len) // stride + 1` sets the hyper-state grid height and the token width `d_model * patch_num`.

Other hyperparameters: preset defaults in `configs/models/TimePro.toml`; tune generically.

## Differences

- Independent rewrite after inspecting the official `model/TimePro.py` and `layers/TimePro_EncDec.py` (no LICENSE file, so the card omits `code` provenance); no source copied.
- The custom CUDA selective scan is replaced by a pure-PyTorch recurrence for `d_state=1` (the paper's default); other `d_state` values are rejected.
- The CUDA-only `DCNv4` deformable convolution for hyper-state mixing is replaced by a fixed-receptive-field depthwise 2-D convolution; this drops learned sampling offsets and is expected to reduce fidelity to the reported numbers.
- Stochastic depth (`DropPath`) is omitted.
