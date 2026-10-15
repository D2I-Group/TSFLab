---
name: "RAGC"
description: "Scalable traffic GNN: per-node window embedding plus node and calendar embeddings, an efficient cosine graph operator linear in node count, stochastic shared-embedding regularization, and residual-difference blocks. Use for large road or sensor networks; not for non-graph or few-channel data."
---

# RAGC

## Idea

- Each node's lookback window is embedded by one linear layer and concatenated with a learned node embedding and time-of-day and day-of-week embeddings read from the calendar covariates.
- `EfficientCosineGraphConv` (Efficient Cosine Operator, ECO) builds a gated, cosine-normalized support from node embeddings and diffuses without materializing a dense adjacency, so cost is linear in the node count.
- `StochasticSharedEmbedding` (SSE) regularizes the node embedding during training (`sse_p`, `use_sse`); this is the "regularized" in the name.
- Each block subtracts the graph-smoothed signal from the hidden state and accumulates it in a skip path; two linear heads (`regression_layer`, `regression_skip`) are summed.
- `graph_regularization_loss` (Laplacian smoothness on `adj_mx`) is exposed as an optional ablation but is not called by `forward` or the trainer.

## When to use

- Large-scale road or sensor networks where quadratic graph convolution is too expensive: the adaptive graph is learned from node embeddings at linear cost.
- Traffic-like data with strong time-of-day and day-of-week effects: calendar embeddings enter every node.
- Not for data without node structure or with few channels; no static graph is needed (it is used only by the optional ablation loss).

## Configure

- `enc_in` follows the node count: must equal the number of nodes (`num_nodes` and `adj_mx` are injected by the runner).
- `input_dim` follows the spatiotemporal features: value plus time-of-day and day-of-week (3); the input must provide at least this many.
- `num_time_in_day` follows the sampling frequency: number of time-of-day slots (official: steps per day, e.g. 96 or 288; default 24).

Other hyperparameters: preset defaults in `configs/models/RAGC.toml` (smoke-scale widths); tune generically.

## Differences

Independent rewrite from arXiv:2506.07179 after inspecting `models/RAGC.py` and `main.py` of the official repository (no license file); nothing copied. SSE, ECO, residual-difference wiring and dual heads match the official `forward`.

- The Laplacian loss (`--lap_sm`, enabled by no released training command) is a separate method, not wired into training.
- Only the ECO (`cos_gconv`) path; the softmax and static-adjacency ablation branches are not implemented.
- ECO output dropout defaults to 0.0 (official 0.1); preset widths are smoke-scale (official `spatial_dim=64`, 4-8 layers).
