---
name: "FEDformer"
description: "Encoder-decoder Transformer with Fourier-enhanced blocks and frequency cross-attention on a few selected modes, inside progressive seasonal-trend decomposition. Use for long-term forecasting of series with a sparse Fourier representation and a global trend; not for strongly irregular or noise-like series."
---

# FEDformer

## Idea

- `FrequencyEnhancedBlock` replaces self-attention: rFFT, a learned per-head complex kernel on a selected subset of modes (`mode_select` low or random), then irFFT (Eqs. 3-4).
- `FrequencyEnhancedAttention` does decoder cross-attention on the selected Fourier modes of queries, keys and values with a tanh score (Eqs. 6-7).
- `series_decomposition` follows every encoder and decoder sub-layer; the decoder accumulates three projected trend updates per layer on a mean-initialized trend (shared `decomposition_encdec` scaffold with `autoformer`).
- Channels are mixed in the value embedding (`forecast_embedding`, with calendar marks).

## When to use

- Designed for long-term forecasting where decomposition captures the global profile (trend) and a sparse set of Fourier modes captures the seasonal detail.
- Linear complexity in sequence length (fixed number of modes), cheaper than full attention on long inputs.
- Channels are embedded jointly; on many weakly correlated channels, channel-independent models are often preferable.

## Configure

- `enc_in`, `dec_in`, `c_out`: number of channels; all three must match.

Other hyperparameters: preset defaults in `configs/models/FEDformer.toml`; tune generically.

## Differences

Clean-room implementation; `MAZiqing/FEDformer` at `c0f6b972` (MIT) is reference only, nothing copied.

- Only the Fourier variant is implemented (no wavelet), with head-local complex kernels and deterministic random mode sets.
- Inputs `[B, seq_len, enc_in]` with optional six-column marks; outputs `[B, pred_len, c_out]`.
- No checkpoint or published-metric reference comparison is claimed.

Citation: Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., Jin, R. "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting." ICML 2022, PMLR 162, pp. 27268-27286. arXiv:2201.12740.
