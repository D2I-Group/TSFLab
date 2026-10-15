---
name: "MTSMixer"
description: "Attention-free MLP mixer with factorized temporal mixing over interleaved subsequences and a low-rank channel bottleneck, under RevIN. Use for multivariate forecasting with redundant, correlated channels on a tight budget; not for exogenous inputs, univariate data, or probabilistic output."
---

# MTSMixer

## Idea

- `TemporalSubsequenceMixer` splits the time axis into `sampling` interleaved subsequences and gives each its own MLP (`fac_T`), replacing attention for temporal dependence.
- `ChannelInteraction` mixes channels through a narrow `d_ff` bottleneck (`fac_C`), exploiting redundancy among channels.
- `FactorizedMixerBlock` adds temporal and channel residuals in sequence with optional LayerNorm.
- `channel_wise_linear` maps `seq_len` to `pred_len` after the blocks; `revin` (no affine) normalizes the input.

## When to use

- Multivariate data whose channels are redundant or low-rank (few principal components): the factorized channel bottleneck mixes them cheaply.
- Tight compute budgets: MLP-only, no attention; the paper reports higher efficiency than Transformer forecasters.
- Not for univariate data with `fac_C` on (the rank must be below the channel count), exogenous or calendar inputs (marks are ignored), or quantile output (point forecast only).

## Configure

- `enc_in`: must equal the dataset's channel count.
- `d_ff`: channel-bottleneck rank; with `fac_C = true` it must be smaller than `enc_in`.
- `sampling`: number of interleaved temporal subsequences; at most `seq_len` (with `fac_T = true`).

Other hyperparameters: preset defaults in `configs/models/MTSMixer.toml`; tune generically.

## Differences

- Local implementation from paper Eqs. (3), (6), and (8); the unlicensed reference repository (`models/MTSMixer.py`) was inspected at the pinned revision, nothing copied.
- Default variant only: equidistant interleaved subsequences, independent temporal MLPs, low-rank channel bottleneck, residual composition, RevIN, and a direct history-to-horizon projection. Attention and random-matrix variants and SVD/NMF refinement are omitted.
- GELU, pre-LayerNorm, and the forecast-only runtime are local choices.
- `FactorizedMixerBlock` is not the cataloged `mixer_block`: it normalizes over channels only, groups temporal mixing by subsequence (Eq. 6), and uses a low-rank channel bottleneck (Eq. 8), so it stays model-local.
