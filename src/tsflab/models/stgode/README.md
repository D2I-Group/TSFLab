---
name: "STGODE"
description: "Tensor-ODE blocks that integrate graph propagation and dilated temporal convolution over a spatial and a semantic graph, fused by a gate. Use for traffic node forecasting on a road graph where long-range spatial correlations matter; not for data without a graph or probabilistic output."
---

# STGODE

## Idea

- Shallow GNNs miss long-range spatial correlations and ignore semantic (non-adjacent but similar) connections; a tensor ODE allows deep propagation, and a semantic graph adds those connections.
- `TensorODEBlock` integrates a derivative of graph propagation, a temporal convolution, and a node-state term with learnable step size (explicit Euler steps), giving deep propagation without stacking GNN layers.
- Two `ODEBranch`es run in parallel: one on the row-normalised physical adjacency, one on a semantic graph from adjacency profiles; both wrap the ODE in dilated causal convolutions.
- A sigmoid gate fuses the branches and a linear layer maps the flattened history to the horizon.

## When to use

- Traffic flow on a road network where distant nodes influence each other through multi-hop paths.
- Requires a dataset adjacency; the semantic graph is derived from it, not from series similarity.
- Point forecasts only.

## Configure

- `enc_in` follows the node count; `num_nodes` and the adjacency come from the dataset graph.
- `input_dim` follows the loader's input features (value plus calendar covariates).
- Other hyperparameters: preset defaults in `configs/models/STGODE.toml`; tune generically (`ode_steps` sets integration depth).

## Differences

Clean-room rewrite from the paper equations; the BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0) is reference-only and was not copied.

- The catalog contract has no training-series DTW matrix, so the semantic graph comes from graph-neighbourhood profiles (`A A^T`, row-normalised) instead of DTW similarity.
- Explicit Euler steps with configurable depth replace the `torchdiffeq` solver; official preprocessing and published-metric comparison are not reproduced.

Citation: Fang, Long, Song, Xie, "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting", KDD 2021, arXiv:2106.12931.
