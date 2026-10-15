---
name: "GCformer"
description: "Global convolution over the whole lookback plus a local channel-independent PatchTST, fused by cross-attention. Use for long-lookback, long-horizon multivariate forecasting where long windows carry signal; not for short windows or probabilistic output."
---

# GCformer

## Idea

- Global branch (Eqs. 1-3, SGConv-style `MultiScaleGlobalConv`): per head and variate, `1 + ceil(log2(L / 32))` 32-tap sub-kernels are upsampled, decay-weighted, concatenated, normalised and split into causal and anti-causal halves; the whole window is convolved by FFT, then a `D` skip, GELU and a head-to-variate linear map; `global_head` maps the window to the horizon.
- Local branch (Eq. 9): a channel-independent PatchTST on the last `context_len` steps (end-padded patches, learnable positions, BatchNorm encoder with residual attention scores, flatten head).
- `CrossFusion` decoder (Eqs. 10-12): both forecasts become hidden tokens and attend to each other in both directions, mixed by `atten_bias`; a channel view (tokens are horizon steps) and a token view (tokens are variates) are mixed by `tc_bias`.
- Learnable shortcuts `global_bias * z_global + local_bias * z_local` are added to the fused forecast; RevIN (or last-value subtraction) wraps the model.

## When to use

- Designed for long-term multivariate forecasting with long input windows: the global convolution covers the full lookback at FFT cost, the patch branch keeps recent local detail.
- Suited when long windows carry signal beyond what a patch Transformer over a short context sees.
- Not for very short windows (the global kernel has fewer than 32 taps) or probabilistic output.

## Configure

- `enc_in`: the dataset's channel count (global kernels are per variate).
- `context_len`: local-branch window, `patch_len <= context_len <= seq_len` (default `seq_len`).
- `patch_len`: local patch length, at most `context_len`.

Other hyperparameters: preset defaults in `configs/models/GCformer.toml`; tune generically.

## Differences

- Independent rewrite of Section 3 (Eqs. 1-3, 8-15, Fig. 3) after reading the pinned official code (no license file, `NOASSERTION`); the `GConv` `cat_randn` mode, PatchTST settings, decoder residual/normalization and shortcut init follow the code.
- The decoder uses full softmax attention without a mask per Eq. (12); the official ProbSparse form is `decoder_attention = "prob"`, with the official head-layout bug corrected.
- The `GConv` sub-kernel count is clamped to at least one for windows shorter than 32 steps; its kernel norm is fixed at construction (same values).
- Not implemented: alternative global branches (`FNO`, `Film`), the local decomposition option, unused official modules, auxiliary forward outputs.

Full detail: `reference.md`.
