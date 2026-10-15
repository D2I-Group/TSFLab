---
name: "TSMixer"
description: "All-MLP Mixer that alternates time-mixing and feature-mixing blocks, then projects the lookback to the horizon. Use as a light baseline on multivariate data with cross-channel information; not for strongly drifting levels (no instance normalization) or tasks needing covariates."
---

# TSMixer

## Idea

- Each `MixerBlock` applies a pre-normalized residual MLP along the time axis and a two-layer residual MLP along the feature axis.
- Stacked `e_layers` times, then `ChannelWiseLinear` maps `seq_len` to `pred_len` on the time axis.
- Pure MLP, no attention; this version applies no instance normalization.

## When to use

- Feature mixing lets channels inform each other, which the paper argues pays off when cross-variate information exists.
- Cheap and simple: a baseline for tight compute budgets.
- No instance normalization, so level shifts between train and test are not corrected; prefer models with `revin` on non-stationary data.
- The auxiliary/static-feature extension of the paper is not implemented, so known covariates are not used.

## Configure

- `enc_in`: must equal the dataset channel count (the feature-mixing MLP is sized by it).

Other hyperparameters: preset defaults in `configs/models/TSMixer.toml`; tune generically.

## Differences

- Clean-room rewrite of the basic historical-target TSMixer from Appendix B.3.1-B.3.2; the Google Research source was a reference only.
- Uses sample-wise two-dimensional LayerNorm and GELU.
- Omits the auxiliary/static-feature extension, dataset-global preprocessing, benchmark-specific BatchNorm, and the training stack.
