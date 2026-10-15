---
name: "HimNet"
description: "Graph-GRU encoder-decoder whose node-specific filters come from hierarchical node, calendar and horizon meta embeddings. Use for spatiotemporal node forecasting (traffic-style sensors) with heterogeneous nodes and daily/weekly cycles; not for data without timestamps or with a known graph to exploit."
---

# HimNet

## Idea

- `_meta` concatenates a node embedding, time-of-day and day-of-week embeddings and a horizon embedding, then projects them to a per-node meta vector (spatiotemporal heterogeneity learned as embeddings).
- The shared `node_adaptive_graph_conv` component builds a graph from that vector (`adaptive_node_embedding_adjacency`) and generates node-specific weights and biases from weight banks (meta-parameter pools) indexed by it, with polynomial order `cheb_k`.
- `MetaGraphGRUCell`s (shared `graph_conv_gru` gating) form the encoder and an autoregressive decoder that feeds back its own prediction at each horizon step.
- The supplied `adj_mx` is ignored; topology is learned only from the embeddings.

## When to use

- Designed for spatiotemporal node forecasting (the paper uses five traffic-style benchmarks) where nodes and times behave heterogeneously and per-node, per-time parameters help.
- Needs calendar marks: time-of-day and day-of-week drive the meta embeddings (without future marks the decoder falls back to a step counter and weekday 0).
- Not for tasks where a known road graph should be used (it is ignored); decoding is step by step, so cost grows with the horizon.

## Configure

- `enc_in`: number of nodes `N` (one value per node; `num_nodes` is injected by the runner).
- `steps_per_day`: samples per day at the dataset's frequency (288 for 5-minute data, 24 for hourly); sizes the time-of-day embedding.

Other hyperparameters: preset defaults in `configs/models/HimNet.toml`; tune generically.

## Differences

- Local rewrite after reviewing the paper and the pinned official BasicTS code; no paper or code problem is recorded.
- Exposes one value per node (`output_dim = 1`); input features are the value plus time-of-day and day-of-week (`input_dim = 3`).
- `adj_mx` and `use_teacher_forcing` are accepted but unused: the decoder always feeds back its own predictions.
