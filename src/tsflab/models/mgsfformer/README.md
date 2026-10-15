---
name: "MGSFformer"
description: "Multi-granularity spatiotemporal Transformer: residual de-redundant coarse views, temporal then cross-station attention, and dynamic fusion of per-granularity forecasts. Use for multi-station air quality or similar sensor networks with strong cross-station correlation; not for independent channels or short windows."
---

# MGSFformer

## Idea

- `_granularity` builds averaged-and-upsampled copies of the embedded series at factors 1, 3, 6, 12 and 24; `ResidualDeRedundant` subtracts what each coarser level can be predicted from the next finer one.
- `SpatioTemporalAttention` applies temporal attention per station and then spatial attention across stations at each step, once per granularity.
- Each granularity has its own flatten-and-linear head and `DynamicFusion` combines the forecasts with sample- and node-specific softmax weights.
- `revin` wraps the model.

## When to use

- Designed for air quality prediction from monitoring stations: patterns differ across sampling granularities and stations are strongly correlated in space and time.
- Spatial attention learns station relations without a supplied graph.
- Not for weakly correlated or independent channels, where cross-station attention adds noise and cost.
- Uses historical targets only (no covariates or calendar marks), and needs `seq_len` of at least 24.

## Configure

- `enc_in`: number of stations (channels); spatial attention runs across them.
- `seq_len` (task): must be a positive multiple of 24 because of the granularity factors 1, 3, 6, 12, 24.

Other hyperparameters: preset defaults in `configs/models/MGSFformer.toml`; tune generically.

## Differences

- Clean-room implementation; the unlicensed author repository is reference-only and its source was not copied.
- Keeps the five granularities, residual de-redundancy, temporal/spatial attention, dynamic fusion, and RevIN.
- Private preprocessing and auxiliary objectives are omitted; only historical targets are consumed.
