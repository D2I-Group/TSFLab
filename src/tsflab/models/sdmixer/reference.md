# SDMixer — reference

## Paper

arXiv 2602.23581 (2026-02); accepted by the DSFA track of PAKDD 2026, later withdrawn from the formal proceedings by the author for lack of institutional funding.

## Differences in detail

No source was copied from the official repository (`https://github.com/SDMixer/SDMixer`, revision `c330c01fa01cdcb3dbf2cde6b0a73a2bbd80a08a`); it ships no LICENSE file (`NOASSERTION`) and is reference-only. Its model file (`models/SDMixer.py`, alongside `layers/RevIN.py`) was read only to resolve structural ambiguities the paper leaves open.

- **Forecast horizon (completeness fix).** The pinned `forward` never reaches `pred_len`: `ForecastHead`/`linear3` are defined but unused, and it returns `[batch, seq_len, enc_in]`. This implementation adds the linear head over the sequence axis implied by Eq. 6 (`Y_hat in R^{B x L' x C}`).
- **Spectral decomposition (Eqs. 3-5).** The pinned `DFT_series_decomp.forward` sets `freq[0] = 0` (zeroing the first batch row, not the DC bin) and applies a single global `topk` threshold across the tensor. Here: an independent top-k magnitude mask over frequency for every (batch, channel) pair, `irfft` for the season, `trend = x - season`.
- **Sparse temporal gate axis (Eq. 8).** The pinned `SparseTopK` sparsifies along the sequence axis; the paper keeps the "top-k channels ... at each time step". The paper is followed.
- **Sparse cross-mixer (Eqs. 12-13).** The pinned block calls `nn.MultiheadAttention` with no sparsification, and its commented-out trend/season MLP branches (`linear1`/`linear2`/`weight`) are dead code. Here: manual scaled dot-product attention (query from trend, key/value from the frequency branch), top-k sparsification of the softmax map, and a sigmoid-gated residual onto the trend branch.
- **Undocumented widths.** `MLP_T` (Eq. 10) is a two-layer GELU MLP over time with configurable `d_ff`; `Q/K/V` (Eq. 12) stay at channel width `enc_in`, consistent with the `sqrt(C)` scale. These are design choices, not paper values.
- **No renormalization after sparsification.** `alpha = TopK(Softmax(...))` is literal, so `alpha` rows do not sum to one.
- Marks and decoder arguments are accepted and unused: SDMixer is a channel-preserving, calendar-feature-free point forecaster in both paper and code.
- No probabilistic output, pretrained artifacts, or checkpoint/metric comparison against an official training recipe.

## Component decisions

Only `revin` is reused (per-instance mean/variance standardization with affine scale/bias, restored around the head output). Other candidates were rejected:

- `dominant_periods` returns detected periods and amplitude weights (TimesNet-style); Eqs. 3-5 reconstruct a seasonal signal from a top-k magnitude mask, so `SpectralDecomposition` is local.
- `series_decomposition` is a moving-average split, a different operation.
- `frequency_band_sampler` (Dualformer) selects depth-indexed contiguous bands across stacked layers; SDMixer has one full-spectrum linear "Enhance" branch (`FrequencyFlow`).
- `harmonic_energy_gate` (Dualformer) computes a spectral-energy ratio; SDMixer's fusion gate is a learned scalar through a sigmoid (`gamma` in `SparseCrossMixer`).
- `sparse_connection_router` (LSINet) is an input-independent Gumbel-softmax router; SDMixer's gates (Eqs. 8, 12) are input-dependent top-k masks (`_topk_mask`).

No new shared component was extracted: these blocks have one consumer, below the two-consumer threshold of curate-components.

## Citation

SDMixer: Sparse Dual-Mixer for Time Series Forecasting, arXiv:2602.23581, 2026.
