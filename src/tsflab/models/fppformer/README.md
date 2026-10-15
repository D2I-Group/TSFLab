---
name: "FPPformer"
description: "Channel-independent patch Transformer with a bottom-up patch-merging encoder using diagonal-masked element/patch attention and a top-down patch-splitting decoder that cross-attends first. Use for univariate-style forecasting of series with outliers; not for cross-channel modelling."
---

# FPPformer

## Idea

- Encoder stage: diagonal-masked element-wise attention inside each patch, then diagonal-masked patch-wise attention, so every element or patch is rebuilt only from the others and outliers are pulled toward the general level.
- Bottom-up pyramid: adjacent patches merge (patch size doubles) after each stage; every stage map is kept as a lateral output.
- Decoder stage: patch-wise cross-attention to the same-resolution encoder map comes before element-wise self-attention, so future queries (position embeddings only) are initialized from the input first.
- Top-down pyramid: the decoder starts at the coarsest patch size and splits patches stage by stage; the output is a linear projection of the final encoder map plus the projected decoder output, inside non-affine `revin`.

## When to use

- Designed to strengthen the Transformer decoder and to be robust to outliers in the input (diagonal masking).
- Hierarchical patches capture structure at several resolutions within a fixed lookback.
- Channel-independent; the cross-variable variant is not implemented.

## Configure

- `enc_in`: number of channels.
- `patch_size`, `num_stages`: `seq_len` must be a multiple of `patch_size * 2^(num_stages - 1)` with at least two top-stage patches.

Other hyperparameters: preset defaults in `configs/models/FPPformer.toml`; tune generically.

## Differences

Independent rewrite of Section IV, Figs. 2-5 and Table I after reading `OrigamiSL/FPPformer` at `49a82073` (Apache-2.0); nothing copied.

- `FPPformer-Cross` (official `--Cross`) is not implemented.
- The official no-op merge/split of the last stage is skipped, relaxing its even-count requirement.
- The position table covers `seq_len + pred_len` steps instead of a fixed 5000-step buffer (same values).
- Training uses the configured loss (paper and official loop: MSE + MAE, Adam, learning rate 1e-4 halved every epoch, batch 16, patience 1). Details in `reference.md`.
