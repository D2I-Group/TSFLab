---
name: "DecompRevisit"
description: "Dual-MLP decomposition forecaster: a moving-average trend forecast under RevIN plus an unnormalized seasonal shift-MLP fed the raw window, summed. Use for non-stationary series with a clear trend and stable seasonality; not for exploiting cross-channel dependence or probabilistic output."
---

# DecompRevisit

## Idea

- A moving-average split gives trend and seasonal parts (Eqs. 1-2).
- Only the trend is instance-normalized (affine `revin`), mapped over time by a three-layer ReLU MLP, and denormalized (Eq. 8).
- The seasonal part is never normalized: `ShiftMLP` (RMSM) embeds it, concatenates the raw input window as context, and maps to the horizon (Eqs. 10-12); `seasonal_head="mlp"` (RMM) reuses the trend MLP instead.
- Trend and seasonal forecasts are summed (Eq. 14).

## When to use

- Series whose level drifts (normalizing the trend handles non-stationarity) while the seasonal amplitude carries information that normalization would erase.
- Cheap MLPs; channels share weights and are forecast independently; point output only.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/DecompRevisit.toml`; tune generically (`kernel_size` odd).

## Differences

- Independent rewrite of RMSM and RMM after reading the pinned official code (no license); nothing copied.
- The shift MLP ends in a linear layer as in the code, not the ReLU of Eq. (12).
- The DR- variants with other seasonal backbones and the MoE/FFT decompositions are not implemented. Detail in reference.md.
