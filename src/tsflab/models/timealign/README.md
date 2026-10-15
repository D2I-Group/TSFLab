---
name: "TimeAlign"
description: "Patch-MLP forecaster trained with a future-reconstruction branch and local/global alignment of history to future representations; inference is a plain patch MLP. Use for long-term forecasting where history and future distributions or frequency content differ; not for cross-channel structure."
---

# TimeAlign

## Idea

- The predictor is a `PatchMLPBranch`: `patch_num` non-overlapping patches, linear embedding, residual MLP layers, flatten decoder, wrapped in `revin` (non-affine).
- A second `PatchMLPBranch` reconstructs the future from the target during training; `DistributionAlignment` pulls each projected history state toward the matching detached future state with local (cosine) and global (relation-matrix) terms.
- `training_objective` returns `L_pred + w_recon * L_recon + w_align * L_align`; validation, early stopping and `forward` use only the history predictor, so inference costs a plain patch MLP.

## When to use

- Long-term forecasting where the input history and the future target differ in distribution; the paper attributes the gains mainly to correcting frequency mismatches between history and future.
- Lightweight settings: the deployed model is a small channel-independent patch MLP; the reconstruction branch only adds training cost.
- Not when cross-channel interactions or covariates drive the target, or for probabilistic output.

## Configure

- `enc_in`: number of channels (sizes RevIN).
- `patch_num`: number of patches; must divide both `seq_len` and `pred_len` (patch length `seq_len / patch_num` for history and `pred_len / patch_num` for the future, so tokens align one to one).

Other hyperparameters: preset defaults in `configs/models/TimeAlign.toml`; tune generically.

## Differences

- Clean-room implementation; the unlicensed repository is link-only and nothing was copied.
- Paper mapping: patched encoders → `PatchMLPBranch`; local/global alignment → `DistributionAlignment`; prediction, reconstruction and alignment objective → `training_objective`.
- The paper's plug-in use with alternative base forecasters, dataset recipes and a numerical comparison with reported results are not reproduced.
