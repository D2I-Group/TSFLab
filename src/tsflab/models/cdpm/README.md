---
name: "CDPM"
description: "Decoupled forecaster: moving-average split, a linear plus signed-sqrt polynomial trend head, and a DDIM seasonal denoiser conditioned on historical patch statistics. Use for series with clear trend and seasonal structure; not for calibrated probabilistic output or tight inference budgets."
---

# CDPM

## Idea

- History statistics normalize both windows; a moving-average decomposition splits trend and seasonal parts.
- Trend (PTM): `Linear(trend)` and `Linear(sign(trend) sqrt(|trend|))` mixed by a learnable `Linear(2, 1)`.
- Seasonal (CDSM): an AdaLN MLP denoiser over the noisy seasonal future, blended with a prior drawn from per-patch historical means and stds.
- Training predicts the clean seasonal future from a noised one (teacher forcing); inference runs deterministic DDIM over all `K` steps.

## When to use

- Series with a distinct trend plus seasonal component, where a nonlinear (square-root) trend term helps.
- Returns one sampled point forecast, not quantiles; inference costs `diffusion_steps` denoiser passes.

## Configure

- `enc_in`: the dataset's channel count.
- `patch_len`: `pred_len` must be a multiple of it, and `seq_len >= patch_len`.

Other hyperparameters: preset defaults in `configs/models/CDPM.toml`; tune generically (`emb_dim` even, `moving_avg` odd).

## Differences

- Independent rewrite from the paper and the pinned official code (`zjt-gpu/CDPM@1d943bb0`, no license file); the official code does not build at that revision.
- Separate `Linear_origin`/`Linear_root` follow the paper; their mix is the code's `Linear(2, 1)` with bias; the signed square root applies elementwise (the code fills only one time step).
- The cosine schedule, `[-1, 1]` clipping and DDIM with `eta = 0` come from the code; the remaining conflicts are recorded as `issues` in `card.toml`.
