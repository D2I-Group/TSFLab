---
name: "SOFTS"
description: "MLP forecaster over inverted series tokens whose channels interact through STAR, a softmax-pooled global core redistributed to every series in linear channel cost. Use for multivariate data with many correlated channels under a compute budget; not for univariate series or probabilistic output."
---

# SOFTS

## Idea

- Channel independence resists drift but ignores correlations; attention or mixers capture them at high cost and depend on each channel's quality. STAR centralizes channel interaction instead.
- `history_embedding` embeds each series' whole lookback into one token (inverted view), so blocks operate across series.
- `SeriesCoreFusion` (STAR) scores per-series core candidates, softmax-aggregates them into one global core, and concatenates it back onto every series through an MLP, in linear channel complexity.
- `SOFTSBlock` adds pre-norm residuals around STAR and a feed-forward layer; `forecast_head` maps each token to `pred_len`; inputs are standardized per window (`use_norm`).

## When to use

- Multivariate data with correlated channels, especially many channels where pairwise channel attention is too expensive.
- Robustness to noisy individual channels: the shared core reduces reliance on any one series.
- Time is handled by one linear embedding of the whole window per series, so fine intra-window temporal structure is not modelled explicitly. Point forecasts only.

## Configure

- No data-dependent parameter: the channel count is read from the input (`enc_in` is validated positive but does not shape any layer).
- Other hyperparameters: preset defaults in `configs/models/SOFTS.toml`; tune generically.

## Differences

Clean-room rewrite; the official `Secilia-Cxy/SOFTS@f5d35fd` (MIT) is reference-only and was not copied.

- `SeriesCoreFusion.aggregate` implements series-to-core aggregation and `forward` the core redistribution.
- Forecast-only rewrite; no numerical comparison against reference outputs is claimed.

Citation: Han, Chen, Ye, Zhan, "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion", NeurIPS 2024, arXiv:2404.14197.
