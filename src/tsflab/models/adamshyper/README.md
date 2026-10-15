---
name: "AdaMSHyper"
description: "Per-scale learned hypergraphs over a conv pyramid, node/hyperedge constraint loss, intra-scale and hyperedge attention. Use for multivariate forecasting with interacting patterns at several temporal scales; not for very short lookbacks or probabilistic output."
---

# AdaMSHyper

## Idea

- Strided `ScaleConv` layers (Conv1d, BatchNorm, ELU; window equals stride) build node sets of length `L`, `L/w1`, `L/(w1 w2)`.
- Per scale, an incidence matrix is learned with `adaptive_node_embedding_adjacency` (`softmax(relu(E_node E_hyper^T))`), top-`eta` per node, threshold `beta`, binary in the forward value with a straight-through gradient.
- Hyperedge features are member-node means; a node constraint and a margin-`gamma` hyperedge constraint form `L_const`, added as `aux_loss` while training.
- Nodes are updated by attention-enriched, degree-normalized hypergraph convolution; hyperedges of all scales exchange information by one self-attention; one Linear maps all node and hyperedge tokens to the horizon.

## When to use

- Series whose time steps interact in groups at several resolutions (multi-scale pyramid, learned hyperedges across time steps).
- Channels form the feature vector of every node, so channel information is mixed; better suited to correlated channel sets than to many independent channels.
- The hypergraph is learned over time, not over channels: it does not need or use a spatial graph.
- Needs `seq_len` long enough for the aggregation windows; point output only.

## Configure

- `enc_in`: number of channels (the hypergraph feature width).
- `window_size`: the pyramid strides; `seq_len // prod(window_size)` must be at least 1. `hyper_num` needs one entry per scale (`len(window_size) + 1`).

Other hyperparameters: preset defaults in `configs/models/AdaMSHyper.toml`; tune generically.

## Differences

Follows the paper (Eq. 3-16); the official code (no license file) was consulted for structure only.

- Head: the paper's single linear over node and hyperedge tokens; the official input skip, hard-coded zero padding to 80, and extra `pred_len` linear are not used.
- Incidence: no fixed logit factor 3; a straight-through term restores gradients that the official `torch.where` blocks; empty hyperedges are masked.
- Constraint loss: Eq. 10-13 with a `lambda_balance` mix, training-only `aux_loss`; validation uses the forecast loss.
- Pyramid: Conv1d + BatchNorm + ELU per scale instead of the official bottleneck CSCM; hard-coded sizes replaced by sizes derived from `seq_len` and `hyper_num`.

Full detail in `reference.md`.
