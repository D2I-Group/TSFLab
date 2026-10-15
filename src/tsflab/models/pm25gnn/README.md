---
name: "PM25_GNN"
description: "Graph GRU encoder-decoder for station networks: neighbour messages over a fixed adjacency enter both GRU gates, and the decoder rolls out with known-future covariates. Use for air-quality or other sensor-graph forecasting with a known station graph; not for data without a graph or covariates."
---

# PM25_GNN

## Idea

- `GraphGRUCell` concatenates each node's input and state with its neighbours' (row-normalized `adj_mx` plus self loops) in both gates, around the shared `graph_conv_gru.graph_gru_step` gating.
- The encoder runs over the lookback with station values and time covariates built by `to_spatiotemporal`.
- The decoder rolls out `pred_len` steps, feeding its previous prediction and the known-future covariates (`future_time_features`) into the next step.
- Graph topology is fixed and supplied through `adj_mx` (identity when absent); nothing is learned about the graph.

## When to use

- Station networks (the paper targets PM2.5) where pollutant transport between neighbours and future covariates drive the forecast; the decoder consumes node-structured future covariates when the dataset provides them, otherwise calendar time-of-day and day-of-week.
- Needs a meaningful adjacency; without one the model degrades to independent per-node GRUs.
- Autoregressive rollout: errors compound over long horizons, and there is no input normalization, so scale the data beforehand.

## Configure

- `enc_in` follows the node count: must equal the number of stations.
- `adj_mx` follows the graph: an `enc_in x enc_in` adjacency (self loops added, rows normalized).

Other hyperparameters: preset defaults in `configs/models/PM25_GNN.toml`; tune generically.

## Differences

Independent clean-room rewrite from the paper; the MIT repository is kept as a reference and no source was copied.

- Repository adjacency cannot supply the paper's geographic distance/direction and wind-conditioned transport edge features.
- Shared calendar marks replace the KnowAir meteorological variables when no node-structured covariates are present.
- The paper's data pipeline and published-metric comparison are not claimed.

Cite: Wang, Li, Zhang, Meng, Meng, Gao, "PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting", SIGSPATIAL 2020, pp. 163-166, doi:10.1145/3397536.3422208.
