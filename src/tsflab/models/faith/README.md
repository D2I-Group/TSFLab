---
name: "FAITH"
description: "Multi-scale trend/seasonal split; attention over sampled FFT bins along the channel axis, then the time axis, on the seasonal part, plus a linear trend forecast. Use for multivariate forecasting where channel interactions and periodic structure both matter; not for very many channels or univariate series."
---

# FAITH

## Idea

- `MultiScaleTrend` (Eqs. 1-3): the trend is a learned weighted sum of edge-padded moving averages (kernels 17 and 49, `series_decomposition`); the seasonal part is the residual.
- `FrequencyAttention` (FEM / iFEM, Eqs. 7-19): rFFT along one axis, keep a fixed subset of bins, run one post-norm attention layer on the real and imaginary parts, zero-pad and inverse rFFT.
- `FCTEMBlock` applies it across channels for every time step (FCEM), then across time for every channel (FTEM), on a `[B, N, L, d]` seasonal embedding.
- The seasonal part goes through `flatten_forecast_head`, the trend through a linear map initialised as a mean; two learned weights fuse them inside affine `revin` (Eqs. 4-5).

## When to use

- Designed to capture both inter-channel and temporal dependence in the frequency domain ("two horizons"), suited to correlated channels with periodic components.
- The trend/seasonal split helps when a smooth trend sits under seasonal variation.
- Channel-mixing: with very many channels the per-step channel attention is costly, and it helps little when channels are nearly independent.

## Configure

- `enc_in`: number of channels (FCEM takes bins from the first `(N + 1) // 2` channel-axis rFFT bins).

Other hyperparameters: preset defaults in `configs/models/FAITH.toml`; tune generically.

## Differences

Independent rewrite after reading `LRQ577/FAITH` at `d445cab5` (no license file, recorded `NOASSERTION`); follows the arXiv v1 text (withdrawn after v1; published in Knowledge-Based Systems 309, 112790).

- The random bin subset is drawn once at construction (`bins` buffer), so inference is deterministic; the official code redraws it every forward. With the shipped settings every bin is kept and both coincide.
- The trend learner is the official single linear layer (Eq. 6's activation-free three-layer MLP is the same affine family).
- Official blocks ignore the configured dropout and `d_ff` (effective 0 and 256); those are the defaults here, and exposed.
- With `dropout > 0` it is applied once to the attention output. Details in `reference.md`.
