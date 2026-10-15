---
name: "xPatch"
description: "Non-Transformer dual-stream forecaster: an exponential-smoothing seasonal-trend split feeds a linear MLP trend stream and a patch-CNN seasonal stream, fused by an MLP. Use for channel-independent forecasting of series with trend plus seasonal structure; not for cross-channel dependence or probabilistic output."
---

# xPatch

## Idea

- `ExponentialDecomposition` (EMA, Eq. 2, or Holt double smoothing with `ma_type=dema`) splits the normalized series into trend and seasonal parts with a recurrence instead of a moving-average kernel.
- `LinearTrendStream`: an activation-free pooled, LayerNorm'd linear bottleneck on the trend.
- `NonlinearPatchStream`: seasonal patches are embedded, then depthwise and pointwise Conv1d with batch norm and a residual, and an MLP head.
- `DualStreamForecaster` concatenates both stream forecasts and fuses them with an MLP; channels are folded into the batch and `revin` wraps the model.

## When to use

- Designed for series that separate into a smooth trend and a seasonal remainder; the EMA trend adapts to recent levels without a fixed window.
- Channel-independent with shared weights, as the paper studies patching and channel independence without attention.
- Lightweight MLP/CNN streams.
- Point output only; the paper's arctangent loss and sigmoid learning-rate schedule are training policies not built in.

## Configure

- `enc_in`: must equal the dataset channel count.
- `patch_len`, `stride`: seasonal patches over `seq_len`; patch count is `1 + (seq_len + pad - patch_len) // stride` with `pad = stride` under `padding_patch = "end"` (last-value replication); with `"none"`, `patch_len <= seq_len`.

Other hyperparameters: preset defaults in `configs/models/xPatch.toml`; tune generically.

## Differences

- Independent, device-neutral rewrite from the paper: the official EMA moves its weights to CUDA unconditionally, which blocked a CPU reference comparison.
- Hidden widths are explicit local defaults because the paper does not specify every dimension.
- End padding is last-value replication.
- The optional `dema` route is the standard Holt level-and-trend recurrence controlled by `alpha` and `beta`, not the paper's default EMA experiment.
- The default loss is MSE; the arctangent loss and sigmoid schedule are not embedded.
