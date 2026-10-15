---
name: "FreTS"
description: "Complex-valued MLPs applied in the frequency domain across channels and across time, with a two-layer direct forecast head. Use as a simple, low-complexity multivariate baseline that learns global dependencies; not for very long lookbacks with many channels (the flatten head grows with both)."
---

# FreTS

## Idea

- Embeds each scalar by a learned vector (`dimension_extension`), giving a `[batch, channels, time, embed]` tensor.
- `FrequencyChannelLearner` and `FrequencyTemporalLearner` each do rFFT, a full complex-matrix MLP (`ComplexFrequencyMLP`, ReLU plus softshrink sparsity) and irFFT, over the channel axis and the time axis.
- The channel learner is skipped when `channel_independence` is set or `enc_in < 3` (degenerate real spectrum).
- A residual connection and a flatten, two-layer LeakyReLU MLP produce the horizon directly.

## When to use

- Designed as a simple MLP alternative to RNN/GNN/Transformer forecasters: the spectrum gives MLPs a global view, and energy compaction focuses them on few key frequency components.
- The channel learner models inter-series dependence; set `channel_independence` when channels are unrelated.

## Configure

- `enc_in`: number of channels; the channel learner needs `enc_in >= 3`.

Other hyperparameters: preset defaults in `configs/models/FreTS.toml`; tune generically.

## Differences

Clean-room implementation; `aikunyi/FreTS` at `6de28ab1` (Apache-2.0) is reference only, nothing copied.

- Eqs. 1-4: orthonormal FFT conversion plus channel and temporal learners; Eq. 5: two-layer direct head; Eqs. 6-7: full complex-matrix MLPs (not diagonal-only).
- Channel learning is omitted when requested or with fewer than three channels.
- Short-term task heads, official recipes, checkpoints and numerical reference comparison are not claimed.

Citation: Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., An, N., Lian, D., Cao, L., Niu, Z. "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting." NeurIPS 2023. arXiv:2311.06184.
