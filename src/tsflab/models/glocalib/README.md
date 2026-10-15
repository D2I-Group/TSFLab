---
name: "GlocalIB"
description: "Forecasting adaptation of the Glocal Information Bottleneck imputation method: a variational per-step encoder regularized by KL and masked-view latent alignment, with a linear forecaster. Use when inputs may be partly masked or missing; not for graph or covariate tasks."
---

# GlocalIB

## Idea

- `_VariationalSequenceEncoder` maps each time step's channel vector to a diagonal-Gaussian latent (reparameterized in training, mean at inference) with an analytic KL term to a standard normal (Eq. 8).
- In training a copy of the window with randomly masked time steps (`mask_ratio`) is encoded, and a `projector` of its mean is aligned to the stop-gradient clean latent (`cos_align`, or `contrastive` InfoNCE over time steps).
- `kl_weight * KL + align_weight * alignment` is exposed as `aux_loss` on top of the forecasting loss; evaluation is a plain single forward.
- The forecast is a linear map over time then a linear value decoder, with `revin` (no affine) around it.

## When to use

- The paper targets imputation under missingness; its masked-view alignment trains latents that stay consistent when parts of the input are masked, which suits inputs with gaps or corrupted steps.
- The KL bottleneck regularizes a small linear forecaster.
- Not an imputation model in TSFLab (forecasting only), and not for graph, calendar or covariate tasks.

## Configure

- `enc_in`: the dataset's channel count (each time step's channel vector is encoded to one latent).

Other hyperparameters: preset defaults in `configs/models/GlocalIB.toml`; tune generically.

## Differences

- Independent rewrite from Eqs. (6)-(8) and (12)-(14); the unlicensed official repository (`NOASSERTION`) was inspected at the pinned revision (`Glocal_IB/Glocal_IB.py`) as reference only; nothing copied, no PyPOTS/PyGrinder dependency.
- Forecasting adaptation: the clean lookback is the anchor that produces the forecast and the detached alignment target; a training-only randomly masked copy is the corrupted view pulled toward it. The imputation setting and its missingness masks are not used.
- No imputation benchmark or published-result comparison.
