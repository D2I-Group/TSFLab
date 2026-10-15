---
name: "STGCN"
description: "Fully convolutional spatio-temporal graph network: gated temporal convolutions sandwich Chebyshev graph convolutions over a fixed adjacency. Use for traffic and sensor-network node forecasting with a known road graph and fast training; not for data without a meaningful graph or probabilistic output."
---

# STGCN

## Idea

- Traffic flow is highly nonlinear with spatial and temporal dependencies; STGCN formulates forecasting on a graph with purely convolutional structures, for faster training with fewer parameters than recurrent models.
- Each `SpatioTemporalBlock` sandwiches a Chebyshev graph convolution (`ChebyshevGraphConvolution`, supports from `graph_spectral.chebyshev_supports`) between two GLU-gated causal convolutions (`TemporalGate`).
- Two blocks with LayerNorm and dropout, then a flatten MLP maps the whole history to the horizon per node.
- Consumes value plus time-of-day/day-of-week features via `to_spatiotemporal` and a fixed adjacency.

## When to use

- Road or sensor networks with a known, fixed adjacency (spectral graph convolution over that graph).
- Short input windows and fast training budgets (no recurrence, no attention).
- The graph is static; changing inter-node relations are not modelled. Point forecasts only.

## Configure

- `enc_in` follows the node count; `num_nodes` and `adj_mx` come from the dataset graph.
- `input_dim` follows the loader's input features (value plus calendar covariates).
- Other hyperparameters: preset defaults in `configs/models/STGCN.toml`; tune generically (only the GLU / Chebyshev path is supported).

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0).

- Each block follows temporal GLU, fixed Chebyshev graph convolution, temporal GLU, using the injected adjacency and the shared `graph_spectral` support builder.
- Only `act_func = "glu"` with `graph_conv_type = "cheb_graph_conv"` is implemented (the paper path).

Citation: Yu, Yin, Zhu, "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting", IJCAI 2018, arXiv:1709.04875.
