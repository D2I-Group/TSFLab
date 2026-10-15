---
name: "inception_block"
description: "TimesNet-style 2-D inception block: mean of num_kernels same-padded square Conv2d branches (sizes 1, 3, ..., 2K-1). Use for periodic series folded into 2-D period images (TimesNet, WFTNet, DRAGON, Times2D); not for 1-D sequences, causal kernels, or concatenated branches."
---

# inception_block

## What it does

`InceptionBlock2d(in_channels, out_channels, num_kernels, init_weight=True)` maps
`x [B, C_in, H, W]` to

`y = (1 / K) * sum_{i=0}^{K-1} Conv2d_{(2i+1) x (2i+1), pad i}(x)`, with `K = num_kernels`,

so every branch keeps the spatial size and the branches are averaged (not
concatenated). Consumers fold a 1D series into a 2D image (rows = cycles, columns
= phase, or time x derivative order) and stack two blocks with a GELU between.

## When to use

Use for series with clear (possibly several) periods once a model folds them
into 2-D views (rows = cycles, columns = phase, as in `timesnet`, `dragon`,
`wftnet`) or builds other 2-D maps such as time by derivative order (`times2d`);
the branches see variation at several scales at once with a fixed parameter
budget. Do not use on 1-D sequences, when kernels must be causal or anisotropic,
or when branch outputs must be concatenated or bottlenecked (`sdgfnet` keeps its
own 1-D dilated "InceptionBlock", a different operator).

## Interface

`InceptionBlock2d(in_channels: int, out_channels: int, num_kernels: int, init_weight: bool = True)`

- `in_channels`, `out_channels` (int >= 1): `Conv2d` channel counts.
- `num_kernels` (int >= 1): number of branches; branch `i` has a square kernel
  `2i + 1` and padding `i`.
- `init_weight`: `True` re-initializes each kernel after construction with
  `kaiming_normal_(mode="fan_out", nonlinearity="relu")` and a zero bias (the
  Time-Series-Library behaviour); `False` keeps the PyTorch `Conv2d` default
  initialization and consumes no extra random numbers.
- `forward(x [B, in_channels, H, W]) -> [B, out_channels, H, W]`; float input, no
  explicit validation (errors come from `Conv2d`).
- State-dict keys: `kernels.<i>.weight` (`[out, in, 2i+1, 2i+1]`) and
  `kernels.<i>.bias` (`[out]`) for `i < num_kernels`. No buffers, no state.
