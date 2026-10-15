---
name: "STTN"
description: "Spatial-temporal Transformer: per-step spatial attention across nodes gated against a fixed adjacency, then per-node temporal self-attention, stacked. Use for node-level traffic forecasting with directed, changing spatial dependencies and longer horizons; not for non-graph data or probabilistic output."
---

# STTN

## Idea

- `SpatialTransformer` computes dynamic multi-head attention across nodes at every time step and gates it against a fixed row-normalized adjacency projection with a sigmoid, so spatial links are learned but anchored to the graph.
- `TemporalTransformer` applies self-attention over time per node; both use post-norm residual blocks with feed-forward layers (`SpatialTemporalBlock`).
- Input is the value plus `cov_dim` covariate/calendar features, embedded with a learnable positional parameter; a flatten linear layer maps to the horizon.

## When to use

- Traffic-style node forecasting where spatial dependencies are directed and change with real-time conditions; the paper targets long-term horizons on PeMS-Bay and PeMSD7(M).
- A road adjacency helps (stationary path); without one an all-ones graph is used and only attention carries structure.
- Attention over all nodes at every step costs `O(T * N^2)`; very large node sets are expensive.
- Not for non-graph multivariate data or probabilistic output.

## Configure

- `enc_in`: number of nodes; input must be `[B, seq_len, enc_in]`.
- `adj_mx`: dataset adjacency `[enc_in, enc_in]`, injected by the runner and row-normalized; all-ones when absent.
- `cov_dim`: number of covariate features taken from the marks (2 = time-of-day and day-of-week for calendar stamps; node covariates on spatio-temporal datasets); missing features are zero-padded.

Other hyperparameters: preset defaults in `configs/models/STTN.toml`; tune generically.

## Differences

- Clean-room implementation from the paper; the unlicensed official repository is reference-only and nothing was copied.
- Covers directed multi-head spatial attention (four heads), the gated stationary graph path, bidirectional temporal attention, and stacked ST blocks.
- The official data pipeline and a metric comparison against the official code are not claimed.
