---
name: "MoGU"
description: "Ensemble of Gaussian iTransformer experts gated by inverse predicted variance, reporting mean plus total (aleatoric + epistemic) variance. Use for multivariate forecasting that needs calibrated per-step uncertainty with correlated channels; not for point-only tasks on tight budgets or calendar-driven data."
---

# MoGU

## Idea

- Each expert (`GaussianITransformerExpert`, Eq. 15) is an iTransformer mean path (inverted variate tokens, post-norm encoder, linear horizon projection) plus `UncertaintyHead`, a softplus variance per variable and horizon step (Eq. 14).
- `precision_weights` is the uncertainty gate of Eq. (10): `w_i = sigma_i^-2 / sum_j sigma_j^-2`, elementwise; there is no gating network.
- `mixture_moments` gives the Eq. (7) mean `sum_i w_i y_i` and the Eq. (12) variance split into aleatoric `sum_i w_i sigma_i^2` and epistemic `sum_i w_i (y_hat - y_i)^2`; `forward` returns `(loc, sqrt(total variance))` as a `[B, H, C, 2]` Gaussian.
- `training_objective` trains with `mogu_loss`, the Eq. (11) precision-weighted sum of per-expert clamped Gaussian NLLs (Eq. 6), gate not detached; validation and test score the combined Gaussian with the configured `nll_gaussian` criterion.
- `num_experts` independent experts (3 in the paper) share one instance-normalized input.

## When to use

- Forecasts that need a per-step uncertainty estimate, split into aleatoric and epistemic parts, rather than a point value.
- Multivariate data with correlated channels: each expert attends across variate tokens like iTransformer.
- Not when only a point forecast is needed on a tight budget (cost is `num_experts` full iTransformers), when calendar marks matter (ignored), or when a multi-modal predictive density is required (output is one Gaussian by moment matching).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (one variate token per channel).

Other hyperparameters: preset defaults in `configs/models/MoGU.toml`; tune generically.

## Differences

- Independent rewrite of Sections 3.2-3.4 (Eqs. 6-15) and Listing 1; the MIT official code was read at the pinned revision to resolve omissions, nothing copied.
- Calendar marks are not used (the official iTransformer appends `timeF` mark tokens).
- The predictive distribution is a Gaussian with the mixture's mean and total variance, not the full mixture density.
- The variance clamp has zero gradient below `nll_eps` (official `nn.GaussianNLLLoss` keeps the gradient).
- Only the iTransformer expert is provided; PatchTST/DLinear experts and the learned-gate MoE/MoGE baselines are not.
- Variance de-normalization follows the official code by default (`denorm_variance = false`; see issues). Details in reference.md.
