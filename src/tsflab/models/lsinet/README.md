---
name: "LSINet"
description: "Lightweight channel-independent patch MLP that replaces attention with a shared sparse interaction matrix learned from a position-only memory table. Use for efficient long-term forecasting with explicit temporal interactions; not for covariate-driven tasks (marks are ignored)."
---

# LSINet

## Idea

- `SharedSparseConnectionRouter` (`sparse_connection_router`) builds one 0/1 interaction matrix per head over time patches from a learnable memory table, not from the input, with a top-k hard mask (`density`).
- `SparseTemporalInteractionBlock` combines a time-invariant patch-mixing MLP, propagation of per-head values through the shared sparse matrix, a time-updating MLP, and a feature-integration MLP with residuals.
- Patches have a learnable position embedding (`positional_encoding`); channels are folded into the batch so every sample and channel reuses the same connections.
- `FlattenForecastHead` maps patch features to the horizon inside `revin`; the paper's auxiliary sparsity loss is not wired in.

## When to use

- Designed for long-term forecasting where linear/MLP models are competitive but lack explicit temporal interaction; MSIM adds it at MLP cost.
- Low overhead and shared interactions suit tight compute budgets.
- Channel-independent: cross-channel structure is not modelled; marks and decoder inputs are ignored.

## Configure

- `enc_in` / `c_out`: both must equal the channel count (channel-independent).
- `n_patches`: target patch count; `stride = seq_len // n_patches`, `patch_len = 2 * stride` (paper Eq. 9), so keep it well below `seq_len`.

Other hyperparameters: preset defaults in `configs/models/LSINet.toml`; tune generically.

## Differences

- Clean-room implementation; the `NOASSERTION` repository is reference-only and was not copied.
- The adaptive sparse regularization loss (ASRL, Eq. 7) is not wired in; sparsity comes from the top-k hard mask.
- The two-class Bernoulli is a single binary-concrete logit (equivalent).
- Memory-to-head reshape uses a per-head split instead of the official `.view` that mixes patch and head axes.
- Ablation branches (SAM, `resdual_block`) and the variable-decomposition head are omitted.
