---
name: "CosDir"
description: "Direction-aware training loss on a DLinear carrier: base loss plus 1 - cosine between forecast and target horizon difference vectors, fixed or uncertainty-weighted. Use when the direction of change matters and series vary in scale; not for changing architecture capacity or probabilistic output."
---

# CosDir

## Idea

- Changes the training loss, not the architecture: `forward` is a plain `dlinear` forecast and the method lives in `training_objective`.
- `horizon_differences` forms per-channel difference vectors along the horizon (first step against the last observed value) for forecast and target.
- `direction_penalty` (Eq. 2) is the mean of `1 - cos(dY_hat, dY)`; it depends only on the direction of change, so small moves keep a gradient.
- `combine` gives `L_base + dir_lambda * L_dir` or, with `weighting = "uw"`, uncertainty weighting with two learnable log-variances (Eq. 3).

## When to use

- Tasks where getting the direction and shape of change right matters, including series whose local scale varies (the penalty is scale invariant).
- The carrier is DLinear (trend/seasonal split), so it suits the same trend-dominated data DLinear does.
- Validation and test metrics use plain `forward`; point forecasts only.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/CosDir.toml`; tune generically (`kernel_size` odd, `dir_lambda`, `weighting`).

## Differences

- Independent rewrite after reading the pinned MIT official code; nothing copied.
- Carrier fixed to DLinear (the paper tests 15 backbones).
- Cosine denominator follows the paper (`||a|| ||b|| + eps`), not the code's per-norm clamp.
- The trainer validates on the configured criterion, not the combined loss; weight decay also applies to the UW log-variances. Detail in reference.md.
