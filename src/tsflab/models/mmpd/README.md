---
name: "MMPD"
description: "Multi-mode patch diffusion: a denoiser conditioned on future-patch tokens, trained with a diffusion-plus-anchor loss; returns an anchor point forecast and can sample trajectories. Use for forecasting with diverse or multi-modal futures; not for cross-channel interaction or exogenous inputs."
---

# MMPD

## Idea

- `FuturePatchBackbone` turns history patches into one token per future patch by cross-attending learned future queries to the patch embeddings.
- `PatchConsistentDenoiser` predicts noise for noisy future patches, conditioned on the token, the diffusion step, and the left and right neighbouring patches, through AdaLN MLP blocks (`AdaLNMLPBlock`).
- `diffusion_loss` mixes the noise-prediction loss with a deterministic anchor term (weight `diffusion_weight`); it is the model's `training_objective`, replacing the configured criterion during training only. Validation and test score the anchor forecast with the standard loss.
- `forward` returns the efficient anchor-step point forecast; `sample` draws multi-mode trajectories.
- Channels are flattened into the batch; series are standardized with detached mean and standard deviation.

## When to use

- Futures with several plausible outcomes, where an MSE-trained single-mode forecast blurs them: the diffusion loss models the future distribution and `sample` exposes diverse trajectories.
- Tasks that need samples as well as a point forecast from one model.
- Multivariate data with weakly related channels (channels are handled independently).
- Not when cross-channel interaction carries the signal, when exogenous covariates or calendar marks matter (marks are ignored), or when mode probabilities are required (the GMM mode fitting is not implemented).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `patch_len`: follows `pred_len`; it is clipped to `pred_len`, the horizon is split into `ceil(pred_len / patch_len)` future patches (the last replicate-padded), and the history is left-padded to a multiple of `patch_len`. Prefer a divisor of `pred_len` and `seq_len`.

Other hyperparameters: preset defaults in `configs/models/MMPD.toml`; tune generically.

## Differences

- Local implementation of diffusion Eq. (3), the Patch Consistent MLP Eq. (7), the anchor term Eq. (8), and AdaLN-MLP Eqs. (12)-(13); official `mmpd_loss.py` and `gaussian_diffusion.py` inspected at the pinned revision as reference only (no license file upstream).
- The evolving variational-GMM mode fitting (Algorithm 1) and per-mode probabilities are not implemented.
- The patch backbone is a compact local one, not a reproduction of every backbone in the paper.
