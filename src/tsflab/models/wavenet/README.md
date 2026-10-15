---
name: "WaveNet"
description: "Channel-independent stack of gated dilated causal convolutions with residual/skip paths, pooled into a direct multistep linear head under RevIN. Use as a convolutional baseline for local temporal patterns per channel; not for cross-channel dependence, covariates, or probabilistic output."
---

# WaveNet

## Idea

- `GatedCausalLayer` applies tanh/sigmoid gated dilated causal convolutions (`gated_dilated_conv`) with dilations 1, 2, 4, ... per block, plus residual and skip 1x1 convolutions.
- Each channel is a univariate series folded into the batch, so weights are shared across channels.
- Skip outputs are summed, passed through ReLU and a 1x1 convolution, average-pooled, and mapped to the horizon by a linear layer; `revin` wraps the network.

## When to use

- A convolutional baseline for local, short-range temporal structure; the receptive field per block grows as `(kernel_size - 1) * (2^layers - 1) + 1`, and the pooled skip sum summarizes the whole window.
- Channels are modelled independently, suiting weakly related channels.
- Point output only; the paper's autoregressive likelihood is not used.

## Configure

- `enc_in`: must equal the dataset channel count.

Other hyperparameters: preset defaults in `configs/models/WaveNet.toml`; tune generically.

## Differences

- Gated dilated causal layers and residual/skip paths follow the paper; BasicTS was a reference only and was not copied.
- Direct multistep regression replaces the audio softmax output; RevIN is added for forecasting.
- No audio-likelihood training or metric comparison against a reference is claimed.
