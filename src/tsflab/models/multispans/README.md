---
name: "MultiSPANS"
description: "Spatiotemporal Transformer over multi-filter convolution and graph-hop tokens, with spatial attention heads masked by a structural-entropy encoding tree of the road graph. Use for traffic or sensor networks with a known graph; not for data without a graph, very large node sets, or probabilistic output."
---

# MultiSPANS

## Idea

- `MultiFilterConv` (MFCL, Eqs. 4-5): temporal filters of sizes 1, 2, 3, 6 with replicate padding, concatenated, then `gconv_hops` propagations by the row-normalised `D^-1 (A + I)`, every hop concatenated, then tanh.
- `build_encoding_tree` (Eq. 3, Sec. 3.3.3) greedily applies the combine or merge operator with the largest structural-entropy reduction, within `max_tree_height`; `multilevel_masks` turns each non-leaf level into a same-subtree head mask (Eq. 9), adds an adjacency mask, and leaves the other heads global.
- `STBlock` applies a temporal then a spatial `STTransformer` (query-plus-position residual, BatchNorm, ReLU FFN), each in an outer residual BatchNorm; scores are scaled by `1 / sqrt(embed_dim)` and masked by `-1e10` fill.
- `Model.forward` adds Laplacian-eigenvector node encodings (`D_s`) and a sinusoidal time table (`D_t`) to the queries, sums the token embedding and every block output as skips, and decodes with `TransposedConvDecoder`.

## When to use

- Traffic forecasting on a road or sensor network with a predefined adjacency: the graph drives the hop propagation, the Laplacian encodings, and the multi-range spatial masks.
- Short lookbacks and horizons (the official setting and contract task use 12 in, 12 out).
- Covariates can enter as extra input channels (`mark_features`); by default none are used.
- Not without a graph (the identity graph removes the spatial structure), not for very large node sets (the encoding tree costs roughly `O(N^3)` NumPy work at construction, spatial attention is quadratic), and not for quantile output.

## Configure

- `enc_in`: must equal the dataset's node count (the runner injects `num_nodes`).
- `adj_mx`: the dataset's `N x N` adjacency, injected by the runner; the identity is used without one.
- `lape_ratio`: number of Laplacian eigenvectors is `round(N * lape_ratio)` (at least 1), so it scales with the node count.

Other hyperparameters: preset defaults in `configs/models/MultiSPANS.toml`; tune generically. Limits: `embed_dim` divisible by `len(conv_kernels) * (gconv_hops + 1)` and `num_heads`; `num_heads` must exceed `max_tree_height`.

## Differences

- Independent rewrite from the paper and the pinned official code (`SELGroup/MultiSPANS@8b7f1898`, no license file, `NOASSERTION`); nothing copied.
- The encoding tree uses the paper's greedy structural-entropy optimisation, not the official Infomap hierarchy.
- The hierarchical correlation score `S_hier` (Eq. 10) is omitted, as in the official code.
- Paper-code conflicts (key projection, additive masking, position routing, mix-hop transpose, Transformer layer form, no `D_b` encoding) follow the official code; `position_routing = "matched"` gives the Sec. 3.3.2 routing. Each case is listed in the card's issues.
