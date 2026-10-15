---
name: "D2STGNN"
description: "Spatio-temporal GNN that decouples diffusion and inherent signals with an estimation gate and residual backcasts, over static, adaptive and dynamic graphs. Use for traffic-like sensor networks with a road graph and calendar marks; not for data without node structure or long-horizon univariate tasks."
---

# D2STGNN

## Idea

- `DecoupledLayer` splits each hidden state into a diffusion part (multi-step graph propagation) and an inherent part (`GRU` plus temporal `Conv1d`), mixed by a sigmoid estimation gate.
- Four graphs: forward and reverse transition supports of the given adjacency, a self-adaptive graph, and a dynamic graph built from the hidden state.
- Each layer subtracts a backcast of both signals before the next layer and emits a partial forecast; partial forecasts are concatenated and decoded by an MLP.
- Time-of-day and day-of-week embeddings come from calendar marks.

## When to use

- Traffic or similar sensor networks where values both spread over a graph and evolve locally, with a known adjacency and daily/weekly patterns.
- Without an adjacency it falls back to identity plus learned graphs.
- Short horizons on graph data (the runtime contract uses 12 in, 12 out); point output only.

## Configure

- `enc_in`: number of graph nodes N.
- `adj_mx`: the dataset's `[N, N]` adjacency (identity when absent).
- `time_in_day_size`: steps per day (288 for 5-minute data).
- `gap`: `pred_len` must be divisible by it.

Other hyperparameters: preset defaults in `configs/models/D2STGNN.toml`; tune generically.

## Differences

- Local rewrite after reviewing the paper and the pinned official BasicTS code; it keeps hidden-state dynamic graph construction, diffusion/inherent branches, estimation gating, residual backcasts and accumulated partial forecasts.
- Dataset loading and training schedules stay outside the model.
