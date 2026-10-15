---
name: "LatentTSF"
description: "Two-stage latent forecaster: a pretrained, frozen per-step autoencoder lifts observations to latent states that a DLinear backbone forecasts, trained with latent MSE and cosine alignment. Use for multivariate data whose channels share an underlying state; not for budgets that cannot afford autoencoder pretraining."
---

# LatentTSF

## Idea

- `LatentStateAutoencoder` expands each time step's channel vector into a higher-dimensional latent state; it is pretrained on reconstruction (`pretrain`, MAE by default) and then frozen.
- Forecasting happens entirely in latent space with the shared `DLinearBackbone` (`dlinear`, using `series_decomposition`), after a non-affine `LayerNorm` on the latent states.
- Predicted latent states are decoded back to observations by the frozen autoencoder decoder.
- `training_objective` adds latent MSE and cosine alignment to the target's encoded states (`mse_weight`, `cosine_weight`) on top of the forecast MSE.

## When to use

- Noisy, partially observed multivariate series where the paper argues observation-space regression learns shortcut, temporally disordered latents ("latent chaos").
- The autoencoder encodes all channels of a step jointly, so channels should reflect a shared underlying state.
- Needs a separate autoencoder pretraining stage (`ae_train_epochs`); set it to 0 to skip.
- The latent backbone is linear (DLinear), so it inherits DLinear's trend/seasonal split.

## Configure

- `enc_in`: number of data channels (autoencoder observation width).
- `kernel_size`: DLinear moving-average window on latent states (preset 25).

Other hyperparameters: preset defaults in `configs/models/LatentTSF.toml`; tune generically.

## Differences

- Clean-room implementation; reference-only source was not copied.
- Mapping: observation-to-state projection -> `LatentStateAutoencoder`; latent forecasting -> `DLinearBackbone`; Eq. 5 objective -> `ModelSpec.training_objective`; two-stage freezing -> `pretrain`.
- Default autoencoder pretraining is 100 epochs; raise `ae_train_epochs` toward 500 for paper-faithful pretraining.
- Optional perceptual loss, external checkpoints, and numerical reference comparison are not included.
