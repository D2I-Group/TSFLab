---
name: "CARD"
description: "Patch Transformer that blends EMA-aligned temporal-token attention with cross-channel attention over aligned tokens. Use for multivariate forecasting where channels are correlated and channel independence leaves accuracy behind; not for hundreds of channels or strongly shifting levels."
---

# CARD

## Idea

- `ChannelAlignedBlock` runs two attentions per layer: over patch tokens of each channel (queries/keys from `exponential_smooth`, a causal EMA over tokens) and across channels at each patch position.
- A learned sigmoid gate (`blend`) mixes the temporal and cross-channel outputs token by token.
- Overlapping patches (length 16, stride 8) are linearly projected with learned positions; an optional statistics token (`use_statistic`) carries window mean and std.
- A flatten linear head forecasts all channels.

## When to use

- Multivariate series with real dynamical dependence among channels over time, which channel-independent patch Transformers ignore.
- Cross-channel attention is quadratic in the channel count; expensive on very many channels.
- No instance normalization is applied in this implementation, so train/test level shifts are not corrected (the statistics token is optional).
- Point output only.

## Configure

- `enc_in`: number of channels.
- `patch_len`: at most `seq_len` (with `stride`, no padding).

Other hyperparameters: preset defaults in `configs/models/CARD.toml`; tune generically. `d_model` must be divisible by `n_heads`.

## Differences

Clean-room implementation: patch projection, causal EMA token alignment, temporal-token and cross-channel attention, learned token blend, direct forecast head. The reference repository (no license file) is only a provenance link; nothing copied. Omitted: the paper's robust (uncertainty-weighted horizon) training loss and the low-rank attention approximation.
