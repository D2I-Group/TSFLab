---
name: "HDMixer"
description: "Pure-MLP mixer with learned variable-length patches and separate within-patch, across-patch and cross-variable mixers. Use for multivariate forecasting with interacting channels where fixed patches cut peaks or periods; not for univariate or many-channel data with weak correlation."
---

# HDMixer

## Idea

- `LengthExtendablePatcher` (LEP) predicts a center shift and width scale for each nominal patch and samples it from the series by bilinear `grid_sample`, so patch boundaries are learned rather than fixed.
- `HierarchicalDependencyBlock` (HDE) applies separate residual pre-norm `AxisMixer` MLPs over the within-patch axis (short-term), the patch axis (long-term), the variable axis and the embedding axis.
- A flatten-and-linear head maps all patch features to the horizon; `revin` wraps the model.
- The mixers are model-local rather than the shared `mixer_block`, because each axis is mixed on its own.

## When to use

- Designed for multivariate series where fixed-length patches lose boundary information (complete peaks and periods) and where cross-variable interactions matter.
- Pure MLP: cheaper than attention-based patch models.
- The variable mixer is a dense MLP over all channels, so it suits a moderate channel count; with weakly correlated channels set `mix_variable = false`.
- Point output only.

## Configure

- `enc_in`: the dataset's channel count (width of the variable mixer).
- `stride`: the nominal patch count is `seq_len // stride` (at least 1).
- `patch_len`: sampled patch length (> 1); the learned width scales around it within `deform_range`.

Other hyperparameters: preset defaults in `configs/models/HDMixer.toml`; tune generically.

## Differences

- Clean-room implementation of LEP and HDE from the paper; the official repository (no license file, `NOASSERTION`) was used as reference only and not copied.
- The explicit center/width sampler (`tanh` offsets scaled by `deform_range`, bilinear `grid_sample`) is a disclosed local design choice.
- Each `AxisMixer` is a single pre-norm residual MLP applied per axis (local time, patch, variable, channel), not the paired time-then-feature mixing of `mixer_block`, so it stays model-local.
