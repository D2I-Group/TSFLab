---
name: "MegaCRN"
description: "Graph-convolutional GRU encoder-decoder whose graph is generated each step from a trainable meta-node memory bank and blended with the given adjacency. Use for traffic or sensor networks with spatio-temporal heterogeneity and a known or partial graph; not for non-spatial multivariate series."
---

# MegaCRN

## Idea

- Hidden states query a trainable `memory` bank by attention; memory-derived node embeddings give a dynamic meta-graph (`_meta_graph`) that is blended with the supplied adjacency through a learned `graph_mix`.
- `MetaGraphCell` is a GRU (shared `graph_conv_gru.graph_gru_step` gating) whose gates use graph-polynomial features of order `cheb_k`; the graph is refreshed at every encoder step.
- The decoder runs autoregressively, feeding its previous prediction plus future time features and the memory-derived node embedding.

## When to use

- Designed for traffic forecasting on road-sensor graphs, where the paper targets spatio-temporal heterogeneity and non-stationarity (node- and time-specific patterns, anomalous situations).
- Learns graph structure from memory, so it helps when the given adjacency is incomplete; the adjacency acts only as a soft prior.
- Uses future time features in the decoder, so timestamps should carry signal.
- Not for non-spatial multivariate data; the autoregressive GRU decoder is slow for long horizons.

## Configure

- `enc_in`: number of spatial nodes N.
- `adj_mx`: N x N adjacency injected by the runner from the dataset; identity when absent; blended with the meta-graph as a soft prior.

Other hyperparameters: preset defaults in `configs/models/MegaCRN.toml`; tune generically.

## Differences

- Clean-room rewrite from the paper structure; reference-only BasicTS source was not copied.
- The supplied adjacency is a soft prior mixed with the meta-graph through a learned `graph_mix`.
- Contrastive memory losses and curriculum teacher forcing are not implemented.
- The official data pipeline and metric reference comparison are outside this module.
