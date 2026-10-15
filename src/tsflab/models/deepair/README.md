---
name: "DeepAir"
description: "Air-quality station forecaster: target-relative regional aggregation of neighbouring stations feeds five residual fusion MLPs (weather, forecast weather, pollutants, calendar/station meta, holistic). Use for station networks with weather and calendar covariates; not for data without covariates."
---

# DeepAir

## Idea

- A `(target, region, source)` projection turns sparse neighbouring readings into a consistent target-relative regional representation.
- Five `FusionNet` residual MLP branches handle historical weather, forecast weather, secondary pollutant context, meta properties (calendar and station embedding), and a holistic mix.
- Branch outputs are merged per horizon step and station with learnable softmax weights, then a sigmoid (targets assumed min-max scaled).
- Future covariates come from calendar marks via `marks`.

## When to use

- Air-quality or similar station networks where neighbouring stations and weather drive the target, and hourly forecasts over a day or two are needed.
- Needs per-station covariates for history and horizon, and min-max scaled targets (sigmoid output in `(0, 1)`).
- Point forecasts only.

## Configure

- `enc_in`: number of stations N.
- `adj_mx`: station projection `[N, regions, N]` or `[N, N]` adjacency (one region); a default regional projection otherwise.
- `cov_dim`: covariate features per station; history and future widths must match it.

Other hyperparameters: preset defaults in `configs/models/DeepAir.toml`; tune generically.

## Differences

- Local implementation from the paper; the unlicensed official code was inspected at the pinned revision and nothing was copied.
- Regional target history stands in for the secondary-pollutant branch; coordinates, pollutant panels, terrain and min-max statistics are dataset responsibilities.
