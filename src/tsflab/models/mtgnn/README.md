---
name: "MTGNN"
description: "Spatiotemporal GNN that learns a directed sparse graph between series, with mix-hop propagation and gated dilated convolutions. Use for multivariate or sensor-network data whose dependency graph is unknown or only partly known; not for weakly related channels or probabilistic output."
---

# MTGNN

## Idea

- `GraphConstructor` learns an asymmetric top-k adjacency from node embeddings and blends it with the predefined graph (`graph_mix`, `build_adj`).
- `MixHop` propagates over the graph with an initial-state residual (`propalpha`), forward and transposed, inside each `MTGNNLayer`.
- Temporal filtering uses the shared causal `gated_dilated_conv` with dilation growing over layers (`dilation_exponential`).
- `to_spatiotemporal` builds the input from values plus time marks; the head reads the last time step and emits `pred_len` values per node.

## When to use

- Multivariate series whose variables depend on one another but whose graph is not given (the paper's setting); a known road or sensor graph can be supplied as a prior.
- Traffic and other sensor networks with short lookbacks (the contract task uses 12 in, 12 out).
- Calendar stamps enter as extra input features (time of day, day of week).
- Not for weakly correlated channels (graph mixing adds noise) or quantile output (point forecast only); the dense `N x N` graph grows quadratically with the node count.

## Configure

- `enc_in`: must equal the dataset's node count (the runner injects `num_nodes`).
- `adj_mx`: injected by the runner from the dataset graph; without one the identity is used, and with `build_adj = false` the predefined graph is used alone.
- `subgraph_size`: top-k neighbours kept per node; capped at the node count.

Other hyperparameters: preset defaults in `configs/models/MTGNN.toml`; tune generically.

## Differences

- Independent rewrite from the paper equations: antisymmetric directed graph constructor, top-k sparsification, forward and backward mix-hop propagation, causal dilated gated temporal convolutions, skip path, and forecast head; the MIT official code was reference-only.
- One kernel size (3) per layer instead of the paper's dilated-inception kernel bank.
- The supplied adjacency is mixed with the learned graph through a learnable weight (`graph_mix`) as a graph prior.
- The official training and data protocol is not reproduced; published metrics are not reproduction claims.
