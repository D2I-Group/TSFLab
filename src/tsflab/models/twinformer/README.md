---
name: "TwinFormer"
description: "Dual-level Transformer: top-k sparse attention inside patches, mean pooling, top-k attention across patch tokens, then a GRU and a linear head over channel-mixed step embeddings. Use for long-sequence forecasting on correlated channels; not for drifting levels (no instance normalization) or weakly related channels."
---

# TwinFormer

## Idea

- `TopKSparseAttention` (Eqs. 1-4): per head, only the `k` largest logits of each row are kept before the softmax.
- `InformerBlock` (Eqs. 8-9, 12-13): `Z1 = Z + CSA(Z)`, then `Z2 = Z1 + FFN(LayerNorm(Z1))` with a GELU feed-forward.
- Each step is embedded with `Linear(C -> d)` plus sinusoidal positions (Eq. 6), the newest `floor(L / P) * P` steps are split into patches (Eq. 7), a shared Local Informer runs inside every patch, and each patch is mean-pooled to one token (Eq. 10).
- A Global Informer runs across patch tokens; a one-layer GRU reads them (Eqs. 14-17) and `Linear(d -> H * C)` maps its final state to every channel's horizon (Eq. 18).

## When to use

- Designed for long-sequence forecasting, splitting attention into a local (within-patch) and a global (across-patch) level.
- Every step embedding mixes all channels, so it assumes correlated channels.
- No instance normalization in the model; level shifts are handled only by the runner's scaling.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count (input embedding and output head are sized by it).
- `patch_len`: patches are `floor(seq_len / patch_len)` and the oldest `seq_len mod patch_len` steps are dropped; choose a divisor of `seq_len` (`seq_len >= patch_len`).

Other hyperparameters: preset defaults in `configs/models/TwinFormer.toml`; tune generically.

## Differences

- Follows the paper's architecture (mean pooling, one block per level, attention without pre-normalization, one GRU layer, linear head), not the official `I3InformerV2` script variant.
- `top_k = 5` as in the paper (the code uses 2); the code's depth is available through `local_layers`/`global_layers`.
- The code's lag and calendar input features, MAE+MSE loss, warm-up cosine schedule, and gradient clipping are not part of the model; training uses the configured loss and catalog optimizer.
- When `seq_len` is not a multiple of `patch_len` the oldest steps are dropped (the code drops the newest).
- Scaling is fitted on the training split only (the official scripts fit scalers on the whole series).
