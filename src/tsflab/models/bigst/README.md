---
name: "BigST"
description: "Linear-complexity global node attention via positive random features, with node and calendar embeddings. Use for traffic forecasting on large road networks (thousands of nodes) with time-of-day effects; not for small non-spatial multivariate sets."
---

# BigST

## Idea

- Spatial attention over all nodes is linearized with positive features `elu(x)+1`: `phi(Q)(phi(K)^T V) / phi(Q)(phi(K)^T 1)`, cost linear in the node count.
- Queries and keys see each node's history summary plus learned node (`node_source`, `node_target`) and time-of-day/day-of-week embeddings.
- The supplied adjacency enters only as a scaled residual prior (`graph_prior`, `prior_scale`) on the values.
- History is flattened per node and projected linearly; a linear head emits all horizons.

## When to use

- Large-scale spatio-temporal forecasting (up to very many nodes) where quadratic graph attention or convolution is too expensive.
- Learns time-varying global node interactions; a road graph is optional (residual prior only).
- Uses time-of-day and day-of-week embeddings, so it needs calendar marks for daily and weekly effects.
- The paper's pretrained long-history extractor is omitted: only the short `seq_len` window is seen. Point output only.

## Configure

- `enc_in`: number of nodes.
- `tod_size`: steps per day of the dataset (24 hourly, 288 for 5-minute data).
- `input_dim`: 1 value + calendar covariates (3 with time-of-day and day-of-week).
- `adj_mx`: optional `[N, N]` prior.

Other hyperparameters: preset defaults in `configs/models/BigST.toml`; tune generically.

## Differences

Clean-room rewrite from the paper equations; the released artifact `usail-hkust/BigST` (Apache-2.0) was reference only. Omitted: the separately pretrained long-history extractor, the spatial regularization loss, the official data pipeline, and the masked-MAE recipe. The supplied graph is only a residual prior, and the runner owns the objective.
