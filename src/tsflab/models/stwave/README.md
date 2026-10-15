---
name: "STWave"
description: "Traffic graph model that splits each series into Haar low (trend) and high (event) bands, encodes each with temporal attention and sampled spectral graph attention, and fuses them with a calendar gate. Use for node-level traffic forecasting with a road graph and mixed long/short-term patterns; not for non-graph data."
---

# STWave

## Idea

- `wavelet_disentangle` splits each node's lookback into a low-frequency trend and a high-frequency event component with a single-level, same-length Haar reconstruction.
- Separate low and high `DualEncoder` stacks each apply a `TemporalModule` (attention plus gated causal conv) and `SpectralGraphAttention` with Laplacian-eigenvector position encoding.
- Graph attention is masked to adjacency plus a few top-energy sampled queries that attend globally, cutting the quadratic cost.
- `AdaptiveFusion` gates the low and high forecasts with a calendar embedding projected to the horizon, then a linear readout.

## When to use

- Traffic flow with both slow daily trends and short events (e.g. storms, incidents) that benefit from separate low/high-frequency encoders.
- A road adjacency is expected: it masks graph attention and defines the spectral positions; without one a ring graph is used.
- Larger node sets where full graph attention is too costly (query sampling keeps about `log_samples * log2(N)` global queries).
- Calendar marks (time-of-day, day-of-week) drive the fusion gate; not for non-graph multivariate data or probabilistic output.

## Configure

- `enc_in` / `num_nodes`: number of nodes (`num_nodes` injected from the dataset, else `enc_in`); input must be `[B, seq_len, N]`.
- `adj_mx`: dataset adjacency `[N, N]`, injected by the runner; symmetrized for the Laplacian, nonzeros define the attention mask; ring graph when absent.
- `hidden_size`: also the number of Laplacian eigenvectors used as positions; only the first `min(hidden_size, N)` are filled.

Other hyperparameters: preset defaults in `configs/models/STWave.toml`; tune generically.

## Differences

- Clean-room implementation from the paper; no BasicTS source copied.
- Keeps wavelet disentanglement, dual encoders, spectral graph positions, query sampling and adaptive low/high fusion.
- Uses a fixed Haar basis, and omits the paper's auxiliary low-frequency loss and published dataset preprocessing.
- The fusion's calendar term is a linear projection of the lookback calendar embedding, not future marks.
