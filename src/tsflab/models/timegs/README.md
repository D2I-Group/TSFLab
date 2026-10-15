---
name: "TimeGS"
description: "Forecasting as rendering: a UNet on the period-folded history selects and weights fixed 2D Gaussian basis kernels that are splatted onto the horizon so vertical neighbours land one period apart. Use for long-term forecasting of strongly periodic series; not for aperiodic data or cross-channel structure."
---

# TimeGS

## Idea

- 2D variation features (Eqs. 4-6): the RevIN-normalized history is folded into rows of `fold_width`, resized to an image, and encoded per branch by a UNet-style encoder into `G` anchor features.
- A fixed, non-trainable bank of elliptically truncated 2D Gaussian profiles (Eqs. 7-8) is mixed per anchor with softmax weights and MLP intensities into `P` composite kernels (MB-GKG, Eqs. 9-11).
- Kernels are rasterized period-continuously (MP-CCR, Eqs. 12-14): each kernel is flattened row-major at the branch period and centred on its anchor, so 2D vertical neighbours land one period apart on the horizon.
- Channel-adaptive aggregation (Eqs. 15-17) mixes branches and components per channel; `training_objective` is the hybrid MSE/MAE loss of Eq. 18.

## When to use

- Long-term forecasting of series with strong, stable periodicity (one or several periods, one per branch), where the forecast is a superposition of smooth periodic bumps.
- Channel-independent encoding with channel-specific aggregation weights.
- Not for aperiodic or trend-dominated data, or where cross-channel interactions drive the target.

## Configure

- `enc_in`: number of channels (channel-specific aggregation weights).
- `periods`: one branch per entry, each the period (in steps) at which kernels are rasterized; dataset-dependent (official scripts: 24 for ETTh1, 96 for ETTm1, 144 for Weather). 0 or a value longer than the horizon span means the span.
- `fold_width`: row width used to fold the history into an image; the paper folds by the period, the official code hard-codes 24 (preset).
- `pred_len`: kernel rows are capped at `2 * ceil(pred_len / min(periods)) - 1`.

Other hyperparameters: preset defaults in `configs/models/TimeGS.toml`; tune generically.

## Differences

- Independent rewrite (official repository has no license file, `NOASSERTION`).
- One shared folded image at `fold_width` (code) instead of folding by each branch period (paper).
- The Gaussian bank follows the code's sheared-centre, row-truncated construction with an extra `coefficient` axis, not the paper's ellipse test.
- `weight_norm = "official"` divides weights by their row sum before the softmax (code); `"paper"` gives the plain softmax. The temperature 0.1 is from the code.
- The preset follows the ETTh1 horizon-96 script; other datasets and horizons change `periods`, `components`, `n_blocks`, `mse_weight`.
- The official multi-GPU rasterization path and the unused `linear` / `skip_linear` layers are not carried.
- Full detail: reference.md, `## Differences in detail`.
