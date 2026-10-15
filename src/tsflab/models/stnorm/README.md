---
name: "STNorm"
description: "WaveNet-style gated dilated causal convolutions whose layers also see spatially and temporally normalized streams, without an adjacency. Use for multivariate sensor-network data mixing global (shared) and local, high- and low-frequency components; not for probabilistic output or graph-based propagation."
---

# STNorm

## Idea

- Multivariate series come from hybrid systems with global/local (spatial) and low/high-frequency (temporal) impacts; spatial and temporal normalization refine the local and high-frequency components and plug into backbones such as WaveNet.
- Each `NormalizedTemporalLayer` concatenates the raw hidden state with a `SpatialNormalization` stream (statistics across nodes) and a `TemporalNormalization` stream (per-node statistics over batch and time, running averages at eval) before the gated convolution.
- The backbone is a Graph-WaveNet-like stack of tanh/sigmoid gated dilated causal convolutions with residual and skip connections; `adj_mx` is ignored.
- Skip outputs are summed and a 1x1 conv head emits the `pred_len` values at the last time step.

## When to use

- Node or channel sets driven partly by a shared global signal (spatial normalization separates each node's local deviation from it), e.g. traffic networks (METR-LA, PEMS-BAY in the paper).
- No graph needed; cross-node information enters only through the spatial statistics.
- Point forecasts only.

## Configure

- `enc_in` follows the node count; it must equal it exactly.
- `input_dim` follows the loader's input features (value plus calendar covariates).
- Other hyperparameters: preset defaults in `configs/models/STNorm.toml`; tune generically.

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0).

- Spatial and temporal normalization streams are concatenated with the raw hidden state inside a causal dilated temporal backbone with residual and skip paths, as in the reference.
- `tnorm_bool` / `snorm_bool` switch each stream off; `adj_mx` is accepted and ignored.

Citation: Deng, Chen, Jiang, Song, Tsang, "ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting", KDD 2021, doi:10.1145/3447548.3467330.
