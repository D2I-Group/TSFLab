---
name: "ModernTCN"
description: "Pure-conv forecaster: large-kernel depthwise convolutions with variable- and feature-grouped ConvFFNs on patch stems. Use for multivariate long-term forecasting with related channels at convolution cost; not for exogenous or calendar inputs or probabilistic output."
---

# ModernTCN

## Idea

- `LargeKernelDepthwiseConv` applies a large depthwise kernel (default 13) plus an optional small-kernel branch for the paper's training-time reparameterization.
- `ModernTCNBlock` keeps the variable axis and runs two grouped pointwise ConvFFNs, first per variable and then per feature, so the second one mixes variables.
- A patch-embedding stem and optional downsampling stages feed `ModernTCNBackbone`; with `use_multi_scale` the head concatenates all stage outputs.
- `decomposition=True` runs separate seasonal and trend backbones via `series_decomposition` (off by default); `revin` normalizes each backbone.

## When to use

- Multivariate forecasting where cross-variable dependency helps: the feature-grouped ConvFFN mixes variables while temporal convolution stays per variable.
- Long lookbacks and tight compute: large depthwise kernels give a wide receptive field at convolution cost, which the paper reports as a better efficiency/accuracy balance than Transformer and MLP models.
- Not when channels are weakly related and mixing risks overfitting, when exogenous covariates or calendar marks matter (marks are ignored), or when quantiles or distributions are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `patch_size`, `patch_stride`: follow `seq_len`; the stem is unpadded, so `patch_size <= seq_len`, and the stage-0 length `(seq_len - patch_size) // patch_stride + 1` must still be at least `downsample_ratio` for every later stage (otherwise the temporal axis collapses).

Other hyperparameters: preset defaults in `configs/models/ModernTCN.toml`; tune generically.

## Differences

- Clean-room implementation of the paper's patch stem, large-kernel depthwise block, and two grouped ConvFFNs; MIT official code not copied.
- Deployment-time kernel fusion (merging the small kernel into the large one) and the non-forecasting task heads are omitted.
