---
name: "PPM"
description: "Probabilistic MLP: maps the window-normalized history to a Gaussian latent prior, pushes sampled latents forward to horizon trajectories, and trains with a KDE likelihood. Use for quantile or sample forecasts on non-stationary series; not when only cheap point forecasts or cross-channel modelling are needed."
---

# PPM

## Idea

- `Model.prior` (Eqs. 2-3): `Linear(L -> D)`, GELU, and two linear heads give `mu` and `log sigma^2` of a diagonal Gaussian latent for every channel of the window-normalized history.
- `Model.sample` draws `K` latents by reparameterization (Eq. 4); `Model.push_forward` concatenates each draw with a linear embedding of the history and maps it to a horizon trajectory with a two-layer GELU MLP (Eq. 5); samples are de-normalized.
- `kde_nll` (Eqs. 6-8): Gaussian-kernel KDE negative log-likelihood (bandwidth `h = 0.3`, log-sum-exp over samples), averaged over horizon and channels, each window floored at `-25`.
- `training_objective` minimizes `alpha * NLL + MM` (`alpha = 0.1`, 100 draws; Eq. 10), where `mean_matching` is the MSE of the sample mean (Eq. 9).
- `forward` returns linear-interpolated empirical quantiles of 100 draws at the configured levels.

## When to use

- Tasks that need a predictive distribution (quantiles or samples) rather than a point forecast.
- Non-stationary series: every window is standardized by its own statistics before encoding and samples are de-normalized, so level and scale drift are absorbed.
- Not for cross-channel dependence (channels share weights and are processed independently); point metrics use the median quantile, and sampling 100 trajectories costs more than a point model.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

No other data-dependent parameter. Other hyperparameters: preset defaults in `configs/models/PPM.toml`; tune generically.

## Differences

Independent rewrite after reading the pinned official code (`ljl8336/PPM`, `6b4206a`, Apache-2.0); nothing copied.

- Follows the code where it departs from the paper: the push-forward input includes a history embedding, windows are standardized per window, and the `-25` floor applies per window, not per element.
- The forecast is a set of empirical quantiles; catalog point metrics use the median, the paper's MSE/MAE the sample mean.
- Alternative kernels and the conditional backbones of Appendix C.1 are not implemented; training uses the catalog optimizer and schedule.
