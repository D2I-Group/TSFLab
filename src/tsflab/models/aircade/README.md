---
name: "AirCade"
description: "Air-quality Transformer with four-path domain-knowledge attention decoupling AQI-weather causality in past and future. Use for multi-station pollutant forecasting with past and future weather covariates; not for data without covariates or tasks needing a fixed adjacency."
---

# AirCade

## Idea

- `DomainKnowledgeAttention` (DK-MSA) mixes four paths (direct, inverse, and two adaptive-graph attention matrices) with learnable multi-environment intervention masks.
- Cade layers (temporal and spatial) use historical weather as query/key and pollutant values as value to decouple synchronous AQI-weather causality.
- A linear map from history to horizon length feeds Cadi layers that condition on future weather covariates.
- Learnable time and station prompts are concatenated to the value and weather embeddings.

## When to use

- Multi-station air-quality (AQI, pollutant) forecasting where meteorology drives the target and weather covariates are available for history and horizon.
- Spatial dependence is learned by attention over stations; a supplied adjacency is ignored.
- Without node-structured covariates it falls back to two calendar covariates from raw marks, which removes the causal-decoupling motivation.
- Time prompts are sized by `seq_len` and `pred_len`, so the trained model is tied to one window; point output only.

## Configure

- `enc_in`: number of stations.
- `cov_dim`: covariate features per station (`[B, T, N, cov_dim]` marks); 2 when raw six-column calendar marks are converted.

Other hyperparameters: preset defaults in `configs/models/AirCade.toml`; tune generically. `d_model` must divide across heads and exceed `2 * prompt_dim`.

## Differences

Local implementation of the paper's Eq. (1)-(13) after inspecting `src/models/AirCade.py` at the pinned revision; no source copied (the repository has no license file). Eq. (1) maps to prompted embeddings, Eq. (2)-(7) to `DomainKnowledgeAttention`, Eq. (8)-(11) to Cade/Cadi, Eq. (12) to the predictor, Eq. (13) to relaxed multi-environment masks. The runtime accepts unequal history and horizon lengths and learns spatial matrices instead of consuming adjacency.
