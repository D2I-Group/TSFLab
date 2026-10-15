---
name: "DRFormer"
description: "Channel-independent patch Transformer with a dynamic sparse tokenizer (diverse receptive fields), max-pooled multi-scale tokens, group-aware RoPE and deconvolution fusion. Use for long-horizon forecasting where patterns span several temporal scales; not for cross-channel modelling."
---

# DRFormer

## Idea

- Dynamic tokenizer (Sec. 3.2.3-3.2.5): the patch-to-`d_model` weight is multiplied by a binary indicator; output group `i` of `mask_groups` may only use the last `i * P / G` patch positions, so groups see different receptive fields.
- Algorithm 1: during the first `mask_epochs` epochs, each group periodically drops its smallest-magnitude active weights and regrows as many random positions, at a cosine-annealed rate.
- Max-pooled copies of the patch tokens (kernels `2, 4, ..., 2^(sequence_num-1)`) are appended as coarser scales; group-aware RoPE adds intra-group and inter-group rotated score matrices (Eq. 11).
- Each pooled group is upsampled by a transposed convolution and summed onto the base tokens; a flatten head maps to the horizon inside non-affine RevIN.

## When to use

- Designed for long time-series forecasting where useful patterns occur at several temporal scales and receptive-field lengths.
- Channel-independent: no cross-channel interaction.
- Training is stateful (indicator schedule depends on the epoch length); a model trained outside the runner keeps its initial indicator.

## Configure

- `enc_in`: number of channels.
- `patch_len`: must not exceed `seq_len` and must be divisible by `mask_groups` (as must `d_model`).
- `d_model / n_heads` must be an even integer (rotary pairs).

Other hyperparameters: preset defaults in `configs/models/DRFormer.toml`; tune generically.

## Differences

Independent rewrite after reading `ruixindingECNU/DRFormer` at `30dcae62` (no license file, recorded `NOASSERTION`); nothing copied.

- Eq. (11) is implemented (sum of inter- and intra-group score matrices) and Eq. (9) uses the group-index angle; the code instead chains both rotations with a different angle.
- Intra-group positions use the code's base-patch units (Eq. 7 unscaled leaves frequencies near zero).
- Post-norm encoder layout as in the code, not the paper's Eq. (12).
- Bug fixes: `output_padding = 1` when the transposed convolution would lose a position; device-agnostic masks; update interval floored at 1; the indicator is applied inside `forward`.
- `active_ratio` is the code's active share (the paper's `SR` bounds the inactive share; equal at 0.5).
- The code's `1.5 x train_epochs` training with best-checkpoint reload at mask freeze is not reproduced.
