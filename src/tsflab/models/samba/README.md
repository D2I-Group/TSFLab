---
name: "SAMBA"
description: "SDE-Mamba: patch-embedded series encoded by two parallel simplified Mamba stacks, one across time per variate and one across variates per patch, fused by an MLP. Use for multivariate forecasting where both temporal and cross-variate dependencies matter; not for univariate data or when speed matters."
---

# SAMBA

## Idea

- Simplification (Sec. 5.1): `SimplifiedMambaMixer` reuses the `mamba` component's selective SSM but feeds the depthwise convolution output to it without the SiLU activation; the SiLU gate is kept.
- Disentangled dependency encoding (Sec. 5.2): after patch embedding, a cross-time stack (per variate, over patches, with a learnable position term) and a cross-variate stack (per patch, over variates) run in parallel on the same embedding.
- The two outputs are concatenated, mixed by an FFN, flattened and projected to the horizon with `flatten_forecast_head`; `revin` (no affine) wraps the model.

## When to use

- Multivariate data where temporal order and cross-variate dependence both carry signal: the two dependencies are encoded separately so neither interferes with the other.
- `e_layers = 0` or `d_layers = 0` keeps a single branch for ablations.
- Not for univariate data (the cross-variate branch is empty); the pure-PyTorch sequential scan is much slower than the official CUDA kernels.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `patch_len` follows `seq_len`: must not exceed the history padded by one `stride`.

Other hyperparameters: preset defaults in `configs/models/SAMBA.toml`; tune generically.

## Differences

Local rewrite of Sections 5.1-5.2 after inspecting the official MIT code (`490257d`); nothing copied.

- The scan, convolution and gating use the shared pure-PyTorch `mamba` component instead of `mamba_ssm` and `causal_conv1d` CUDA kernels: portable but much slower, and not compared numerically.
- `use_act=True` restores the SiLU after the convolution, as the official kernel path does.
- The preset is smaller than the official per-dataset scripts (e.g. Electricity: `d_model=512`, `d_layers=2`); per-dataset settings are not preset.
