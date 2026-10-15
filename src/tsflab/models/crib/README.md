---
name: "CRIB"
description: "Forecaster for partially observed multivariate series: mask-aware patch embedding, attention over all channel-patch tokens, and a Gaussian information bottleneck with a consistency loss. Use for data with missing values and correlated channels; not for many-channel data."
---

# CRIB

## Idea

- `PatchEmbedding` jointly convolves value and observation-mask patches, so NaNs or an explicit `mask` are handled without imputation; statistics use observed entries only.
- `UnifiedVariateEncoder` flattens all channel-patch tokens into one sequence for self-attention, with variable and temporal embeddings.
- A diagonal Gaussian latent is sampled in training (its mean at evaluation), then an MLP `predictor` forecasts.
- Training adds `aux_loss`: KL compactness plus MSE consistency between the original and a randomly masked/noised view.
- Objective: `IB_weight * MAE(Y_hat, Y) + Consis_weight * MSE(enc_clean, enc_noisy) + KL_weight * KL(q(z|x) || N(0, I))` (defaults 1, 1, 1e-6).

## When to use

- Forecasting directly from incomplete multivariate inputs (NaNs or a same-shaped observation mask), without a separate imputation step.
- Channels interact through attention over every channel-patch token; cost grows with `channels * seq_len / patch_len` tokens, so not for many-channel data.
- Point forecasts at evaluation (the latent mean is used).

## Configure

- `enc_in`: the dataset's channel count.
- `patch_len`: must divide `seq_len`.

Other hyperparameters: preset defaults in `configs/models/CRIB.toml`; tune generically (`model_dim` divisible by `heads_num`).

## Differences

- Local implementation; the official `TSL_models/CRIB.py` and `CRIB_module.py` (no license) were inspected at the pinned revision; nothing was copied.
- Maps Eqs. 3-5 to patch encoding, all-channel/all-patch attention and the predictor; Eqs. 7-9 to the Gaussian bottleneck; Eqs. 11-12 to consistency plus KL `aux_loss`.
- The prediction term is the configured training loss; use MAE for the paper objective.
- Dataset-specific missingness generation is outside the model. Published training schedules, checkpoints and metric reference comparison are not claimed.
