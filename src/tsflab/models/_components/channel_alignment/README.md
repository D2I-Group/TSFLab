---
name: "channel_alignment"
description: "fit_channels: slice or right-pad with zeros the last axis to exactly the requested width; no parameters. Use for matching a dataset's feature count to a fixed cell input width when dropping trailing channels is acceptable; not for cases where every channel matters (use a learned projection) or padding needs a mask."
---

# channel_alignment

## What it does

`fit_channels(values, width)` makes the final axis exactly `width` wide
deterministically: if `values.shape[-1] >= width` it returns
`values[..., :width]` (keeping the leading channels), otherwise it appends
`width - C` zero channels on the right.

## When to use

Use to adapt an input whose feature count may be smaller or larger than a cell's
fixed input width, and where dropping trailing channels is acceptable. Do not
use when all channels matter and the width is smaller (use a learned projection),
or when padding should carry a mask.

## Interface

`fit_channels(values, width)`

- `values`: any tensor with at least one axis; only the last axis is touched.
- `width` (int >= 1): target size. Raises `ValueError("width must be positive")`
  for `width < 1`.
- Returns `[..., width]`. When `C >= width` this is a view slice (never the
  same object, even if `C == width`); when `C < width` it is a new tensor from
  `values.new_zeros` plus `torch.cat`, so dtype and device follow `values`.
- No parameters or state; no gradient for the padded zeros, gradient flows to the
  kept slice.
- Extra channels are silently dropped, which can discard information.
