---
name: "LSTM"
description: "Per-node LSTM with shared weights over value plus calendar or node covariates and a two-layer MLP horizon head. Use as a recurrent baseline for node-structured data with covariates; not for tasks needing cross-node or graph interaction."
---

# LSTM

## Idea

- Each node is treated as its own sequence (nodes folded into the batch), so no graph or cross-node interaction is used.
- `to_spatiotemporal` supplies the value plus `cov_dim` covariates (calendar `[time_in_day, day_in_week]` or node-structured covariates), linearly projected (`input_projection`) before a multi-layer `nn.LSTM`.
- The last hidden state feeds a `Linear` - GELU - `Linear` head that emits all `pred_len` steps directly.

## When to use

- A classic recurrent baseline for spatiotemporal datasets (for example air quality) where calendar or external covariates carry signal.
- Shared weights across nodes keep it small regardless of node count.
- Not for tasks that need a graph or cross-node interaction; no normalization is built in.

## Configure

- `enc_in`: must equal the node/channel count.
- `cov_dim`: covariate features per node — 2 for calendar marks (time of day, day of week), otherwise the dataset's covariate count `F`.

Other hyperparameters: preset defaults in `configs/models/LSTM.toml`; tune generically.

## Differences

- Clean-room implementation of gate-based recurrence, shared per-node encoding, optional covariates, and a direct horizon decoder.
- CauAir (no license file, `NOASSERTION`) is reference-only; nothing was copied.
- Paper-task or checkpoint reference comparison is not claimed.
