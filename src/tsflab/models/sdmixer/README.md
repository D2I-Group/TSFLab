---
name: "SDMixer"
description: "Dual-stream sparse mixer: FFT top-k season/trend split, a channel-sparse time mixer on the trend, spectral enhancement of the season, and sparse cross-attention fusion under RevIN. Use for noisy multivariate data with periodic components and weak or sparse cross-channel links; not for probabilistic output."
---

# SDMixer

## Idea

- Multivariate data often has multi-scale structure, weak correlations, and noise; SDMixer extracts trends in the time domain and local dynamics in the frequency domain, and uses sparsity to filter uninformative cross-variable information.
- `SpectralDecomposition` keeps the top-k amplitude frequency bins per series as the season; the residual is the trend.
- `SparseTemporalFlow` mixes variables with a linear layer, keeps only the largest-magnitude channels per step (`channel_sparse_ratio`), then mixes over time with an MLP.
- `FrequencyFlow` enhances the season with a learned linear layer on the real part of its spectrum.
- `SparseCrossMixer`: trend queries attend to the frequency branch with top-k sparsified weights, plus a sigmoid-gated residual; a linear head maps time to the horizon and `revin` wraps the model.

## When to use

- Series with a few dominant frequencies (top-k spectral season) plus a slower trend.
- Multivariate data where only some channels matter for each other: channel mixing is sparsified per time step instead of dense.
- Calendar marks are ignored; point forecasts only.

## Configure

- `enc_in` and `c_out` follow the dataset channel count; both must equal it (`enc_in == c_out` is enforced).
- `spectral_top_k` follows `seq_len`: at most `seq_len // 2 + 1` (the rFFT bin count); it is the number of dominant frequencies kept as season.
- Other hyperparameters: preset defaults in `configs/models/SDMixer.toml`; tune generically.

## Differences

Clean-room rewrite; official `SDMixer/SDMixer@c330c01` (no license, `NOASSERTION`) read only to resolve ambiguities. Recorded in `card.toml` issues:

- Adds the linear forecast head implied by Eq. 6; the official `forward` never reaches `pred_len`.
- Spectral split follows Eqs. 3-5 per (batch, channel), fixing the official batch-row zeroing and global top-k.
- Sparse gate keeps top-k channels per time step (paper), not along time (code).
- Cross-mixer is the paper's top-k sparsified attention with gated residual, not the official dense `nn.MultiheadAttention`; `alpha` is not renormalized after top-k.
- Unspecified widths: two-layer GELU time MLP with `d_ff`, `Q/K/V` at width `enc_in`.

Full detail and component decisions in `reference.md`.
