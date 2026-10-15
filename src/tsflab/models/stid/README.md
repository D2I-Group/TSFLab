---
name: "STID"
description: "Simple MLP over each node's history plus learnable node, time-of-day, and day-of-week identity embeddings, without a graph. Use for traffic-style node forecasting with strong calendar cycles under a small compute budget; not for exploiting a known graph or for probabilistic output."
---

# STID

## Idea

- Spatio-temporal GNNs grow complex for small gains; the key bottleneck is that samples are indistinguishable in space and time, which identity embeddings fix with a plain MLP.
- Flattens each node's lookback (values plus calendar features) into one vector with a linear layer.
- Concatenates a learnable node embedding and time-of-day and day-of-week embeddings from the latest step, then fuses them with a linear projection.
- Encodes with residual pointwise MLP blocks (`ResidualPointwiseBlock`) and a linear head to the horizon; `adj_mx` is unused.

## When to use

- Sensor networks (traffic, similar node data) where nodes behave differently and daily and weekly cycles dominate.
- Strong efficiency baseline: no graph convolution, no attention, no recurrence.
- Nodes interact only through shared weights; no explicit cross-node propagation. Point forecasts only.

## Configure

- `enc_in` follows the node count; it must equal it exactly (node embedding table).
- `num_time_in_day` follows the sampling frequency: steps per day (24 for hourly, 288 for 5-minute data).
- `input_dim` follows the loader's input features (value plus time-of-day and day-of-week).
- Other hyperparameters: preset defaults in `configs/models/STID.toml`; tune generically.

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0).

- Flattened node histories are combined with node, time-of-day, and day-of-week identities, processed by pointwise residual blocks, and projected directly to the horizon, as in the reference.
- `adj_mx` is accepted and ignored.

Citation: Shao, Zhang, Wang, Wei, Xu, "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting", CIKM 2022, doi:10.1145/3511808.3557702 (arXiv:2208.05233).
