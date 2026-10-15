---
name: "MambaSimple"
description: "Pure-PyTorch Mamba selective state-space stack over per-step channel embeddings with a linear time-to-horizon head. Use for multivariate forecasting with correlated channels and long windows, as a portable SSM baseline; not for many weakly correlated channels or speed-critical runs (no fused kernels)."
---

# MambaSimple

## Idea

- Each time step's channel vector is projected to `d_model` and processed by `MambaResidualBlock`s (`mamba`), the selective state-space scan written in plain PyTorch with no CUDA kernels.
- A final RMSNorm and channel projection restore `enc_in` features, then a `Linear(seq_len, pred_len)` along time produces the horizon.
- Each variate is standardized over its window and restored on the output; time marks are ignored.

## When to use

- Mamba's input-dependent (selective) scan scales linearly with sequence length, so long lookback windows stay tractable in principle.
- All channels share one embedding per time step, so it suits correlated multivariate data.
- Not for many weakly correlated channels (channel mixing at the input), nor when timestamps carry signal (marks are ignored).
- The sequential PyTorch scan is portable but slower than fused-kernel Mamba.

## Configure

- `enc_in`: number of data channels; `c_out` must equal it.

Other hyperparameters: preset defaults in `configs/models/MambaSimple.toml`; tune generically.

## Differences

- Clean-room implementation: the Mamba equations come from the shared pure-PyTorch `mamba` block and the forecast wrapper was designed independently.
- The Time-Series-Library link is reference-only; no source or checkpoint reference comparison is claimed.
- Inputs `[B, seq_len, enc_in]`, outputs `[B, pred_len, enc_in]`; marks are ignored.
