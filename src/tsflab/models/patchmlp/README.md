---
name: "PatchMLP"
description: "Multi-scale patch MLP; a moving average splits the patch embedding into a channel-mixed smooth branch and a channel-independent noisy residual branch. Use for long-horizon multivariate forecasting with useful cross-variable interaction; not for exogenous inputs or probabilistic output."
---

# PatchMLP

## Idea

- `Emb` embeds each channel at four patch scales (default lengths 48, 24, 12, 6, stride half the patch) and concatenates them.
- A moving-average `SeriesDecomp` (`moving_avg=13`) splits the embedding into smooth `X_s = AvgPool(X)` and residual `X_r = X - X_s` parts.
- The smooth branch adds gated inter-variable (channel) mixing after its intra-variable MLP; the noisy residual branch stays channel-independent.
- The branches are summed and `projector` maps `d_model` to `pred_len`; inputs are standardized per window (`use_norm`).

## When to use

- Long-horizon multivariate forecasting where channels share smooth, correlated dynamics (the paper stresses cross-variable interaction) while the noise is channel-specific.
- Series with structure at several local scales, captured by the four patch sizes.
- Efficient alternative to patch Transformers: MLPs only, no attention.
- Not when timestamps or covariates matter (marks are ignored) or when quantiles are needed (point output); the channel-mixing layer is `enc_in x enc_in`, so very many channels cost more.

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (sizes the channel-mixing layers).
- `patch_len`: four patch lengths, each at least 2 and at most `seq_len`; each scale yields `(seq_len - p) / (p // 2) + 1` patches, and `d_model // 4` should be at least that patch count (each patch gets `d_model // 4 // patch_num` features).

Other hyperparameters: preset defaults in `configs/models/PatchMLP.toml`; tune generically.

## Differences

- Local rewrite following the paper's architecture and equations; the authors' repository declares no license, so it is `reference-only`: `model/PatchMLP.py` was inspected at the pinned revision and no source was copied.
- The paper does not fully specify every initialization and dataset-specific setting; no numerical-reference comparison or reported-result claim is made.
- The local `SeriesDecomp`/`MovingAvg` act on the patch-embedded `(B, C, d_model)` feature axis, not the raw time axis of the cataloged `series_decomposition` component, so they stay model-local although the moving-average arithmetic coincides.
