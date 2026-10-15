---
name: "RILoss"
description: "Loss framework on a DLinear carrier: adds an exponential HSIC penalty that pushes forecast residuals to depend on injected noise, i.e. to look like noise. Use to test a residual-informed objective on multichannel targets; not for one-channel targets, where the HSIC term is undefined."
---

# RILoss

## Idea

- RI-Loss changes the objective, not the forecaster: `forward` is a plain shared `dlinear` backbone (moving-average decomposition, seasonal and trend linear maps), one of the paper's five backbones.
- `ri_loss` (Eq. 8, Algorithm 1): observation loss plus `ri_weight * exp(-temperature * HSIC(Y - Y_hat, eps))`, with `eps` fresh uniform noise of the residual's shape, so training maximizes the dependence between residual and noise.
- `hsic` is the biased centred-trace estimator `tr(K H L H) / (n - 1)^2` with Gaussian Gram matrices (bandwidth 1); the kernel samples of one window are its channels, and per-window values are summed over the batch.
- `training_objective` uses the configured criterion (MSE in the paper) as the observation loss; validation and test use the plain criterion.

## When to use

- Noisy multichannel data where a pointwise loss alone overfits noise: the penalty asks residuals to behave like noise rather than carry structure.
- A lightweight carrier (DLinear) suits trend-plus-seasonal data and quick objective comparisons.
- Not for one-channel targets (`features = "MS"` or univariate): HSIC over channels is undefined and the objective raises.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels; the target needs at least two channels for the HSIC term.

Other hyperparameters: preset defaults in `configs/models/RILoss.toml`; tune generically (`kernel_size` odd).

## Differences

Independent rewrite after reading the official code (`shang-xl/RI-Loss`, `55a10ae`, no license file); nothing copied; the forecaster is the catalog `dlinear` component.

- Noise: the paper samples `U(-1, 1)`, the code `U(0, 1)`; the preset follows the code, `noise_low = -1` reproduces the paper (a factor-two noise scale under the distance kernel).
- The biased HSIC estimator of the code is used, not the U-statistic of Eq. 4.
- Only the DLinear carrier is provided (the paper also uses Informer, Autoformer, iTransformer, RAFT); there are no learnable loss parameters despite the title.
