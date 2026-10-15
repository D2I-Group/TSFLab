---
name: "VolDyVAE"
description: "Patch VAE whose forecast distribution has an MLP location and a GRU volatility-state scale that evolves over the horizon, trained by Gaussian NLL plus KL and output as sampled quantiles. Use for probabilistic forecasts with time-varying uncertainty; not for point-only tasks or many weakly related channels."
---

# VolDyVAE

## Idea

- `Model.encode`: RevIN, non-overlapping patches of length `P` over all channels jointly, and an MLP to the mean and log-variance of a diagonal Gaussian latent per patch (Eq. 3).
- `Model.decode_normalized`: a two-layer MLP projects the past latents to `M = ceil(H / P)` future patch latents (Eq. 4); one shared location MLP decodes past and future patches (Eq. 5).
- `VolatilityDynamics` (Eqs. 6-7): a GRU over patch latents gives `sigma = Softplus(Linear(h)) + xi`; its final past state initializes the future scale path, so uncertainty evolves over the horizon.
- `training_loss` (Eq. 8): Gaussian NLL of the reconstruction and of the forecast plus `beta * KL` to `N(0, I)`, installed as `training_objective`.
- `forward` returns empirical quantiles of `num_samples` posterior draws.

## When to use

- Designed for probabilistic forecasting where uncertainty changes over time (volatility clusters, heteroscedastic noise) rather than a static scale.
- Outputs quantiles; catalog point metrics use the median quantile.
- Each patch mixes all channels, so the encoder width grows with `P * C`.
- The scale `sigma` is not RevIN-inverted (as in the paper and code), so it cannot see the window's standard deviation.

## Configure

- `enc_in`: must equal the dataset channel count (patches flatten `patch_len * enc_in` features).
- `patch_len`: history uses `N = ceil(seq_len / P)` patches (the series tail is front-filled when not a multiple) and the horizon `ceil(pred_len / P)` patches cut to `pred_len`; divisors of `seq_len` and `pred_len` avoid both.

Other hyperparameters: preset defaults in `configs/models/VolDyVAE.toml`; tune generically.

## Differences

- Defaults follow the released `VolDy_VAE.yaml` (`dynamic_dim = 128`, `hidden_dim = 256`, `vol_hidden_dim = 128`), not the paper's `D = 256`.
- The location head is a three-layer ReLU MLP (Eq. 5 and code), not the "linear projection" of the implementation details.
- The forecast is empirical quantiles; samples are decoded as one batch (same distribution, different random stream).
- With an `MS` target the forecast NLL uses the target channel only; the reconstruction NLL uses every channel.
- Training uses the catalog optimizer and schedule; the plug-in variants (TimeGAN, K2VAE, PatchTST with a VolDy head) are not implemented.
