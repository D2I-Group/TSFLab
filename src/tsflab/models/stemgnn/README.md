---
name: "StemGNN"
description: "Learns a latent correlation graph between series, then models inter-series and temporal structure in the spectral domain with graph and DFT filters in residual blocks. Use for multivariate data with strong but unknown inter-series correlations; not for weakly correlated channels or probabilistic output."
---

# StemGNN

## Idea

- Captures inter-series correlations (Graph Fourier Transform) and temporal dependencies (Discrete Fourier Transform) jointly in the spectral domain, learning the graph from data instead of a predefined prior.
- `LatentCorrelationGraph`: per-node GRU summary, query/key attention, symmetrised; no predefined adjacency.
- Each `SpectralTemporalBlock` takes Chebyshev graph terms, applies a learned complex filter in the rFFT domain, then inverse FFT and a linear forecast.
- Two blocks are stacked; the first emits a backcast subtracted from the input before the second, and the forecasts are summed then mapped to the horizon.

## When to use

- Multivariate or node data whose series are correlated but the relation graph is unknown.
- Periodic temporal structure that a frequency-domain filter can capture.
- The learned graph is dense over all series, so cost grows quadratically with channel count. Point forecasts only.

## Configure

- `enc_in` follows the node/channel count; it must equal it exactly.
- Other hyperparameters: preset defaults in `configs/models/StemGNN.toml`; tune generically.

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0).

- A latent correlation graph drives Chebyshev graph terms and temporal Fourier filtering in doubly residual spectral-temporal blocks, as in the reference.
- No predefined adjacency is used.

Citation: Cao, Wang, Duan, Zhang, Zhu, Huang, Tong, Xu, Bai, Tong, Zhang, "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting", NeurIPS 2020, arXiv:2103.07719.
