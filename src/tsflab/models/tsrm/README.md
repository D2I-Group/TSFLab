---
name: "TSRM"
description: "Lightweight channel-independent encoder: strided multi-kernel conv representations, attention over them, and transposed-conv merge back to the window, stacked with a global residual. Use for multi-scale temporal patterns on weakly coupled channels under small budgets; not for cross-channel dependence (unless TSRM_IFC)."
---

# TSRM

## Idea

- Representation layer (Eq. 1): `K` 1D convolutions with stride equal to the kernel size, optional dilation, depthwise or grouped, each followed by ELU, concatenated along the sequence axis.
- Two pre-activation residual blocks on the representation: one-group `GroupNorm`, GELU, multi-head self-attention (softmax or sparse `entmax15`); then `GroupNorm`, GELU, a position-wise feed-forward.
- Merge layer (Eq. 2): split into the `K` parts, invert each with a matching transposed convolution back to length `T`, and fuse with `Linear(K d -> d)`.
- A global residual sums representations across the stacked layers.
- Each value is lifted by a shared bias-free `Linear(1 -> d)` inside affine `revin`; `flatten_forecast_head` maps `[T, d]` to the horizon. `inter_feature = true` is TSRM_IFC (feed-forward over all `F x d` features of a position).

## When to use

- Kernels of different sizes and dilations give multi-scale temporal features within one window.
- Channels share weights and are processed independently by default, suiting weakly coupled channels; enable `inter_feature` to mix channels per position.
- Lightweight by design; suits tight compute budgets.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count.
- `conv_layers`: `[kernel, dilation, groups]` per kernel; every convolution must fit `seq_len` (`conv_length(seq_len, kernel, dilation) >= 1`). The paper sizes the largest kernel to cover about 50-80% of the input; the defaults (3/5/10, dilations 1/2/3) come from the shipped ETTh1 setup for `seq_len = 96`. `groups` must be -1 (depthwise) or divide `d_model`.

Other hyperparameters: preset defaults in `configs/models/TSRM.toml`; tune generically.

## Differences

- Follows the official code where it departs from the paper: RevIN before the value lift, no positional embedding, one-group `GroupNorm`, a two-layer ReLU feed-forward, ELU after the RL convolutions.
- Not implemented: probabilistic-sparse attention, pooled representations, the time-feature embedding, imputation and pretraining heads, `n_kernel > 1`, and the ML gradient switch named in the paper.
- Training uses the configured loss and catalog optimizer (the paper uses MSE + MAE, Adam with weight decay, `ReduceLROnPlateau`); no multi-scale grid search.
- The official batch-size-dependent reshapes and dropped last test batch are not reproduced. Reported benchmark numbers are not reproduced.
