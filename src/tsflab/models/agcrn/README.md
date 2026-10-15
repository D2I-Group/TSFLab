---
name: "AGCRN"
description: "Graph GRU with node-embedding-generated filter weights and a self-learned adjacency; needs no predefined graph. Use for traffic-style node forecasting with short horizons and node-specific patterns; not for long-horizon or channel-independent univariate tasks."
---

# AGCRN

## Idea

- Learns a row-normalized adjacency from trainable node embeddings (`adaptive_node_embedding_adjacency`), so no predefined graph is needed.
- Node Adaptive Parameter Learning: `node_adaptive_graph_conv` generates per-node Chebyshev filter weights and biases from the node embeddings through a shared weight bank.
- Plugs these convolutions into GRU gates and candidate state (`graph_conv_gru`), encoding the history recurrently.
- Decodes all horizons at once from the final hidden state plus a learned horizon embedding with an MLP readout.

## When to use

- Spatiotemporal node forecasting (traffic, sensors) where nodes have their own patterns and inter-dependencies, but no reliable predefined graph is available; any supplied `adj_mx` is ignored.
- Uses time-of-day and day-of-week covariates per step when calendar stamps are given.
- Recurrent encoding of short windows (the contract uses 12 in, 12 out); not designed for long lookbacks or long horizons.
- Node embeddings grow with the node count; point output only.

## Configure

- `enc_in`: number of nodes (series).
- `input_dim`: 1 + covariate features per node; 3 with calendar stamps (value, time-of-day, day-of-week).

Other hyperparameters: preset defaults in `configs/models/AGCRN.toml`; tune generically.

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS AGCRN implementation. It keeps node-adaptive parameter generation, the learned row-normalized graph, Chebyshev filtering, and graph-GRU recurrence, and uses the common mark and forecast contracts (calendar stamps become time-of-day and day-of-week features). Only `output_dim = 1` is supported.
