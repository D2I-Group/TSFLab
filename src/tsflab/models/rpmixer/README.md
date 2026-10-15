---
name: "RPMixer"
description: "Graph-free all-MLP mixer for large spatial-temporal data: frequency-domain complex temporal projections plus frozen random projections across nodes in residual blocks. Use for large sensor networks (hundreds of nodes) without needing an adjacency; not for few-channel data or when node outputs need normalization."
---

# RPMixer

## Idea

- `ComplexTemporalProjection` mixes the time axis in the frequency domain with a learnable complex `length x length` matrix (fft, matmul, ifft).
- `FixedRandomProjection` projects across nodes through a frozen, per-block seeded Gaussian matrix to `random_dim`, and `spatial_reconstruction` maps it back, increasing diversity between blocks without an explicit graph.
- `RPMixerBlock` stacks the two with identity residual connections, so each block behaves like a base learner of an ensemble; `channel_wise_linear` decodes `seq_len` to `pred_len`.
- There is no normalization or graph input.

## When to use

- Large spatial-temporal datasets (traffic, sensors) where the paper shows a general forecaster without spatial relations can beat graph models.
- Many correlated nodes: node mixing through random projections is cheap relative to full node attention.
- Not for few-channel data (node mixing has little to work with); there is no instance normalization, so scale the data beforehand.

## Configure

- `enc_in` follows the node count: must equal the number of nodes (channels).

Other hyperparameters: preset defaults in `configs/models/RPMixer.toml`; tune generically.

## Differences

Written for TSFLab from paper Eqs. (1)-(6) after inspecting `src/models/rpmixer.py` of the unlicensed CauAir reference at the pinned revision; no source copied.

- Each block: pre-activation complex FFT temporal projection, a distinct frozen random node projection, learned spatial reconstruction, and exact identity residuals before a shared horizon decoder.
- Uses only observed node values; graph adjacency and timestamp marks are ignored, as the graph-free formulation permits.
- Extra feature construction, MAE training, and benchmark hyperparameters are not reproduced.
- `RPMixerBlock` has no LayerNorm, unlike the cataloged `mixer_block`, so it stays model-local; the decoder reuses `channel_wise_linear`.

Cite: Yeh, Fan, Dai, Saini, Lai, Aboagye, Wang, Chen, Zheng, Zhuang, Wang, Zhang, "RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data", KDD 2024, pp. 3919-3930, doi:10.1145/3637528.3671881.
