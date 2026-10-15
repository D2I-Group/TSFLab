---
name: "PCATransformer"
description: "Training-split PCA compresses the covariates to a few principal scores; a vanilla encoder-decoder Transformer forecasts the target channel. Use for single-target forecasting with many correlated covariates (MS); not for forecasting all channels or probabilistic output."
---

# PCATransformer

## Idea

- Changes the data, not the network ("less is more"): in the `MS` setting the `enc_in - 1` covariates are projected onto their top `n_components` principal axes and concatenated with the untouched target, so the forecaster sees `n_components + 1` channels.
- PCA (centring, SVD, axes by explained variance) is fitted once on the scaled training split in `training_setup`; mean and axes are stored as buffers, not trained.
- `reduce` projects the history and the decoder label window; the zero horizon is appended in the reduced space.
- The forecaster is the vanilla Transformer (full-attention encoder, causal decoder with cross-attention, `DataEmbedding` with calendar marks), projecting to the single target (`c_out = 1`).
- The paper applies the same PCA step to five other backbones; this entry is the PCA + vanilla Transformer member.

## When to use

- Single-target forecasting with exogenous covariates whose information is concentrated in a few principal components (strongly correlated, low effective rank).
- Many covariate channels that would make a full multivariate Transformer slow or overfit; reducing them cuts cost.
- Timestamps carry signal: calendar marks enter through `DataEmbedding`.
- Not for forecasting every channel (`features` must be `MS`; output is `[batch, pred_len, 1]`), for weakly correlated covariates that PCA would blur, or for probabilistic output.

## Configure

- `enc_in`: total channel count including the target (last channel); at least 2.
- `n_components`: number of principal scores kept, in `[1, enc_in - 1]`; the paper selects it per dataset from the covariates' explained variance on the training split (preset 2).
- `freq`: the dataset's sampling frequency, for the `timeF` calendar embedding.

Other hyperparameters: preset defaults in `configs/models/PCATransformer.toml`; tune generically.

## Differences

- Independent rewrite from the paper after reading the pinned official code (MIT); the encoder-decoder uses the catalog's shared Time-Series-Library blocks.
- The official loader fits PCA on the whole standardized series including validation and test rows; here it is fitted on the training split only, removing that leakage.
- Deterministic SVD PCA with a fixed sign convention (scikit-learn may flip score signs); before `training_setup` the projection is a placeholder keeping the first `n_components` covariates.
- Only the vanilla Transformer backbone is provided; runner defaults follow the Time-Series-Library because the official `run.py` is missing. Reported numbers are not reproduction claims. Details in reference.md.
