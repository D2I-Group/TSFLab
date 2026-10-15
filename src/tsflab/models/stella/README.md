---
name: "STELLA"
description: "Lightweight residual MLP over each station's history plus coordinate-based spatial and hour/day/month embeddings, no attention. Use for atmospheric station forecasting (weather, wind) with many stations and calendar effects under a small budget; not for graph-structured or cross-station attention needs."
---

# STELLA

## Idea

- `SpatialEmbedding` (Eq. 3): a two-layer ReLU feed-forward network over each station's standardized 3-D coordinates, so parameters do not grow with the number of stations; without `coords_path` it becomes a learnable per-station table (the paper's RPE ablation).
- `TemporalEmbedding` (Eq. 4): learnable hour-of-day (24), day-of-month (31) and month-of-year (12) tables summed for the first history step of each sample.
- `Model.embed` (Eq. 5): a shared linear map of each station's history to `d_model` plus the spatial and temporal embeddings.
- `ResidualMLP` (Eq. 6): `L` residual Linear-ReLU-Linear-Dropout blocks, then one linear layer to the horizon per station.

## When to use

- Atmospheric time series over many stations (one channel per station), where location and season explain much of the signal.
- Tight parameter budgets (STELLA-10k): no attention, and the coordinate network is independent of station count.
- Stations interact only through shared weights and coordinates, not through a graph or attention. Point forecasts only.

## Configure

- `enc_in` follows the station count; it must equal it exactly.
- `coords_path` follows the station set: a local `.npy` of shape `[stations, 3]` in channel order; without it the learnable per-station table (RPE ablation) is used, which is the default preset.
- Other hyperparameters: preset defaults in `configs/models/STELLA.toml` (official STELLA-10k: `d_model = 32`, two layers, dropout 0.2, MAE); tune generically.

## Differences

Independent rewrite of Sec. 3.3-3.5 (Eqs. 3-6) of arXiv v2 after reading `GestaltCogTeam/STELLA@cdb64e2` (no license, `NOASSERTION`); nothing copied.

- Station coordinates are read from `coords_path` (like the official `pos_data.npy`), since they are not part of the runtime batch; the default preset is the RPE ablation, not the paper's headline configuration.
- Calendar tables use the paper's sizes 24/31/12 (zero-based day and month) from raw marks; the official code allocates 24/32/13 in a dataset-dependent order. With node covariates `[time_in_day, day_in_week]` only the hour term is available.
- Each channel is one station (`C = 1`); multi-variable stations are not separated.
- Theorem 3's input/output parameter count omits the output bias shape; the MLP and STPE terms match. The official LR schedule, weight decay, and clipping are left to the trainer.

Citation: Fu, Wang, Shao, Diao, Wu, An, Yu, Li, Xu, "On the Integration of Spatial-Temporal Knowledge: A Lightweight Approach to Atmospheric Time Series Forecasting", NeurIPS 2025, arXiv:2408.09695.
