---
name: "SCINet"
description: "Original SCI recursive interaction tree with residual temporal projection and optional two-stack refinement. Use for multivariate temporal structure; lookbacks must divide evenly across tree levels."
---

# SCINet

## Idea

- Recursively split even/odd observations; multiplicative scaling and additive/subtractive coupling exchange information.
- Each coupling operator uses replication padding, a configurable convolution, LeakyReLU, dropout, a width-three convolution and Tanh.
- Interleave the recursive outputs, add the input residual and project time to the prediction horizon.
- The optional second stack receives the entire first-stack history plus its forecast.

## When to use

- Multivariate forecasting with temporal dependencies at multiple resolutions.
- Point forecasts; calendar marks are ignored.
- Window lengths must be divisible by `2**num_levels`; for two stacks the concatenated history must also divide evenly.

## Configure

- `enc_in`: channel count.
- `num_levels`: number of recursive splitting levels.
- `hidden_size`: absolute convolution hidden channels; unset means `enc_in` (official hid_size=1).
- `positional_encoding` and `instance_norm`: optional original PE and RIN paths; preset leaves both off.
- Other parameters: `configs/models/SCINet.toml`.

## Differences

Independent implementation aligned to cure-lab/SCINet at `02e6b0af`; see `reference.md`.

- Supports the original standard decoder, groups=1, modified interaction and full-history concatenation; segmented decoder and partial concat variants are not exposed.
- A temporal Linear without bias equals the official kernel-one Conv1d. Two-stack API returns the final forecast; intermediate supervised training is not exposed.
- RIN avoids the official in-place division that breaks input autograd.
- Legacy kernel-five second convolutions, reversed coupling signs, forced PE/normalization and biased projection were corrected; old records are not official-aligned results.
