---
name: "TimeMixer"
description: "Fully MLP multiscale mixer: downsampled scales are decomposed into seasonal and trend parts, mixed bottom-up (seasonal) and top-down (trend), and per-scale predictors are summed. Use for long- or short-term forecasting of series with distinct fine- and coarse-scale patterns; not for very short lookbacks or covariates."
---

# TimeMixer

## Idea

- Average-pools the input into `down_sampling_layers + 1` scales, each with its own `revin` and linear channel embedding.
- `PastDecomposableMixing` (Eqs. 3-5) splits each scale (moving average via `series_decomposition`, or `DFTDecomposition` top-k seasonal with `decomp_method = "dft_decomp"`), mixes seasonal parts fine-to-coarse and trend parts coarse-to-fine with `TemporalMixer` MLPs, and adds a feed-forward residual.
- Future-multipredictor mixing (Eq. 6): each scale has its own temporal and channel linear predictor; the per-scale forecasts are summed.

## When to use

- Series with intricate variations that look different at fine and coarse sampling scales (microscopic seasonal detail vs. macroscopic trend), in long- or short-term forecasting.
- Lookbacks long enough to survive repeated downsampling.
- Channels enter through a linear embedding (channel mixing); not for exogenous covariates or calendar features (marks ignored).

## Configure

- `enc_in`, `c_out`: number of channels; must be equal.
- `down_sampling_window`, `down_sampling_layers`: scale lengths are `seq_len // window^i`; every scale must stay non-empty.
- `top_k` (only for `dft_decomp`): at most half the coarsest scale length.
- `moving_avg`: odd moving-average window of the trend split.

Other hyperparameters: preset defaults in `configs/models/TimeMixer.toml`; tune generically.
The preset follows the ETT and Weather scripts of Time-Series-Library 4e938a1: `d_model` 16, `d_ff` 32,
three down-sampling layers; their learning rate (0.01) and batch size (16) are run settings.

## Differences

- Clean-room implementation; the Apache-2.0 repository is reference-only and was not copied.
- Inputs `[B, seq_len, enc_in]`, outputs `[B, pred_len, c_out]`; marks and decoder arguments are accepted and ignored.
- Omits the official channel-independent and non-forecast branches, official recipes, and checkpoint or metric comparison.
