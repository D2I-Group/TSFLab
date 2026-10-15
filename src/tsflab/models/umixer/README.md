---
name: "UMixer"
description: "Channel-independent U-Net of patch/feature MLP mixers with skip fusion and a stationarity correction that restores the scale lost in deep layers. Use for non-stationary series whose channels have differing distributions; not for tasks that need cross-channel interaction or probabilistic output."
---

# UMixer

## Idea

- `AxisMixer` mixes along the patch axis and the feature axis; down mixers halve the patch count by average pooling, a bottleneck mixer follows, and up mixers interpolate back and fuse the saved skip tensors (`skip_fusion`).
- `StationarityCorrection` rescales the processed features by the ratio of spectral power (latent patch autocorrelation energy) between the patch embeddings and the processed features, gated per feature, to restore non-stationary information.
- Channels share weights; `revin` wraps the model and a flatten linear head predicts the horizon.

## When to use

- Designed for non-stationary series (trends, shifting distributions) where deep layers wash out scale information.
- Channels are processed separately to avoid the influence of distribution differences among them; no cross-channel modelling.
- U-shaped merging of low- and high-level patch features for local temporal dependencies.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count.
- `patch_len`, `stride`: patch count is `ceil((seq_len - patch_len) / stride) + 1` with right zero-padding, then halved per `e_layers` level; choose them so `seq_len - patch_len` is a multiple of `stride`.

Other hyperparameters: preset defaults in `configs/models/UMixer.toml`; tune generically.

## Differences

- Clean-room rewrite from the paper (the reference repository has no license); no source was copied.
- The correction is applied to the latent patch features before the direct forecast head.
- `AxisMixer` is post-norm (residual sum then LayerNorm) with a different mixing order from the cataloged pre-norm `mixer_block`, so it stays model-local.
