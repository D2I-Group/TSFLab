---
name: "CauAir"
description: "Covariate-driven station forecaster: cache-attention over a few learnable coarse regions in linear cost, then propagation through future covariates. Use for many-station air-quality data with weather or calendar covariates; not for data without covariates or tasks needing an explicit graph."
---

# CauAir

## Idea

- `CacheAttention` assigns stations to a few learnable cache slots per head, aggregates values in that coarse view, and broadcasts back, costing O(P*N*d) rather than O(N^2).
- `CachLormer` mixes cache-attention and a `SwiGLU` feed-forward in parallel with softmax-learned weights.
- AQI and weather history are summarized by station-shared linear maps and joined into a past association, refined by `past_cachlormer`.
- A second `future_cachlormer` propagates that association through future covariates (derived from calendar marks) and a linear `decoder` maps to the horizon.

## When to use

- Nationwide or many-station air quality where weather covariates drive the target; cost is linear in station count.
- Needs covariates per station for history and horizon; without weather it falls back to two calendar features.
- Graph-free: any adjacency is ignored, so it cannot exploit a known road or distance graph.
- Point forecasts only.

## Configure

- `enc_in`: number of stations N.
- `cov_dim`: covariate features per station; the model rejects inputs whose historical or future covariate width differs (2 for the calendar fallback).

Other hyperparameters: preset defaults in `configs/models/CauAir.toml`; tune generically (`dim` must be divisible by `heads`).

## Differences

- Local implementation written for TSFLab; the official `src/models/cauair.py` was inspected at the pinned revision and no external source was copied.
- Future covariates are derived from calendar marks when no future weather is provided; adjacency is dropped.
