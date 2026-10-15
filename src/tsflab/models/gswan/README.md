---
name: "GSWaN"
description: "Graph WaveNet whose graph convolution is a node-embedding query-key self-attention over road and adaptive adjacency edges, with a time covariate. Use for sensor-network traffic forecasting with per-pair dynamic dependencies; not for data without nodes or calendar marks."
---

# GSWaN

## Idea

- Spatial Graph Transformer (SGT, Eq. 4): for each support (forward and backward road random walks, plus the adaptive `softmax(relu(e1 e2))` of Eq. 5) it applies one plain diffusion head and `num_heads` attention heads, `order` hops each, and projects the concatenated terms with a 1x1 convolution.
- `pair_attention` (Eq. 6): scores `Mish(K^T Q)` between node pairs, non-edges masked, divided by temperature 7 and normalized over source nodes; keys and queries (Eqs. 7-8) are convolutions over the remaining time axis on `x + e1` and `x + e2`, so each pair's weight depends on the current input and on both sensors' identities.
- `GSWaNLayer` (Fig. 5): valid dilated gated convolution `tanh * sigmoid`, the SGT with a residual, the layer residual, BatchNorm, and a skip read at the last time step.
- `Model` embeds the value and the time covariate with two parallel 1x1 layers, stacks `blocks x layers` layers, sums the skips and decodes all `pred_len` steps with a Mish 1x1 MLP.

## When to use

- Designed for traffic sensor networks (PEMS) where dependencies between sensor pairs change with the current traffic state and each sensor behaves differently.
- Uses a road graph when available and learns an adaptive graph from node embeddings; needs a node-structured dataset.
- Uses a calendar covariate (time of week by default), so timestamps should carry signal.
- Not for long windows: the preset is 12 steps and the attention is quadratic in node pairs.

## Configure

- `enc_in`: the dataset's node count (runner-injected `num_nodes` takes precedence).
- `adj_mx`: road graph, injected by the runner; without it only the adaptive support is used.
- `time_feature`: `time_of_week` (default, official) or `time_of_day`, matching the dominant calendar cycle.

Other hyperparameters: preset defaults in `configs/models/GSWaN.toml`; tune generically.

## Differences

- Independent rewrite of Secs. 3.2.2-3.2.6 (Eqs. 4-8) of arXiv 2302.09956; the official code has no license (`NOASSERTION`) and was read for reference only.
- Eq. 6 multiplies `A * QK^T`; the code (followed) uses `A` only as an edge mask.
- Time covariate: the official loader feeds time-of-week from the row index; here it is computed from calendar marks (`time_of_day` also available).
- Key/query kernel widths derive from the valid dilated lengths (officially hard-coded for 12 steps); the unused last-layer SGT is not built.
- Not reproduced: the whole-series max clamp (leakage), training augmentations, missing-value filling, learning-rate decay and clipping; the catalog trainer and loss are used.

Full detail: `reference.md`.
