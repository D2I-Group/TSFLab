---
name: "GCLSTM"
description: "LSTM whose four gates are Chebyshev spectral graph convolutions of node input and state, with a direct node-wise horizon readout. Use for station networks with a known graph and covariates, such as air quality; not for data without node structure."
---

# GCLSTM

## Idea

- `chebyshev_supports` (`graph_spectral`) turns the adjacency into scaled-Laplacian Chebyshev supports of order `Ks`.
- `GraphConvLSTMCell` computes input, forget, output and candidate gates jointly from a `ChebyshevGraphProjection` of concatenated input and hidden state.
- The cell is unrolled over `seq_len` on value plus calendar/node-covariate channels (fitted to `1 + cov_dim`).
- A linear layer maps the final node state directly to all `pred_len` steps.

## When to use

- Designed for spatiotemporal forecasting over monitoring stations (PM2.5 in the paper) where neighbouring nodes share dynamics through a graph and meteorological or temporal attributes act as node signals.
- Needs a meaningful adjacency; without one it falls back to an all-ones graph and loses its spatial prior.
- Not for ordinary multivariate series without node structure, nor for probabilistic output.

## Configure

- `enc_in`: the number of graph nodes (stations); `adj_mx` must be `enc_in x enc_in`.
- `adj_mx`: the runner's adjacency, converted to scaled-Laplacian Chebyshev supports of order `Ks`.

Other hyperparameters: preset defaults in `configs/models/GCLSTM.toml`; tune generically.

## Differences

- Clean-room implementation from the paper. No author implementation was identified; the recorded unlicensed `PoorOtterBob/CauAir` repository is reference-only and a former derived implementation was deleted.
- The paper's feature pipeline and graph construction are unavailable, so this entry uses the runner's adjacency, the supplied calendar and node covariates, one graph-convolutional recurrent cell, and a direct multi-horizon readout.
- The dataset-specific 72-hour experiment, optimization protocol and published metrics are not reproduced.
