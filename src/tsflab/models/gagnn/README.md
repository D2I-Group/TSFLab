---
name: "GAGNN"
description: "GRU city encoder plus learned soft city-to-group assignment with group-level message passing, for nationwide air quality. Use for node-level forecasting on a graph of correlated, possibly distant sites with calendar covariates; not for data without a node structure or univariate series."
---

# GAGNN

## Idea

- A shared `nn.GRU` encodes each node's history (value plus calendar covariates through `to_spatiotemporal`) into one state per city.
- `GroupAwareLayer` softly assigns cities to learned latent groups, pools them, passes messages between groups through a softmax group affinity, and returns group context to the cities.
- City-graph propagation over the normalized supplied adjacency, group features, and a residual are fused and layer-normalized, repeated for `num_layers`.
- A linear head maps each node state directly to all `pred_len` steps.

## When to use

- Designed for multi-site (city/station) forecasting where geographically distant sites are still strongly correlated: latent groups capture dependencies the spatial graph misses.
- Uses a supplied adjacency plus calendar covariates; without a graph it falls back to identity and relies on the learned groups only.
- No instance normalization: expects the dataset's standard scaling; level shifts are not corrected.
- Target setting is air quality hours to days ahead, not long-horizon benchmarks.

## Configure

- `enc_in`: number of nodes (cities); must match the adjacency size.
- `adj_mx`: the dataset's `[nodes, nodes]` adjacency; self-loops are added and it is normalized; identity when absent.
- `cov_dim`: number of calendar mark features used per step (the first `cov_dim` marks; missing ones are zero-filled).

Other hyperparameters: preset defaults in `configs/models/GAGNN.toml`; tune generically.

## Differences

- Clean-room rewrite from the paper (GRU city encoder, soft group pooling, learned group correlations, city/group/residual fusion); the pinned official code was a reference only, nothing copied.
- Location attributes and the paper's full air-quality feature set are unavailable; the supplied adjacency and calendar covariates are used instead.
- Output is a direct multi-horizon linear head. Published-metric comparison is not claimed.
