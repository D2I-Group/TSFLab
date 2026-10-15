---
name: "BiMamba"
description: "Bidirectional Mamba+ (forget-gated selective scan) over patch tokens, blending independent and mixing channel paths. Use for long-term multivariate forecasting when it is unclear whether channels should be mixed; not for graph or probabilistic tasks."
---

# BiMamba

## Idea

- `MambaPlus` wraps the shared `mamba` selective-scan block with a complementary forget/new-feature gate; `BiMambaPlusEncoder` runs it forward and on the flipped sequence and fuses both.
- Patch tokens (length 16, stride 8) are encoded per channel (`independent_encoder`) and with channels concatenated per patch (`mixing_encoder`).
- `SeriesRelationDecider` turns the average positive soft-rank correlation between channels into a sigmoid gate that blends the two forecasts.
- Flatten heads map patch tokens to the horizon; input is standardized per window and denormalized at the output.

## When to use

- Long-term forecasting where long-range history must be retained (forget-gated scan) at linear cost.
- Datasets that differ in how much inter-series dependence matters: strongly correlated channels lean on the mixing path, weakly correlated ones on the independent path.
- The mixing path grows with the channel count; point output only; marks are ignored.

## Configure

- `enc_in`: number of channels (`c_out` must match).
- `patch_len`: at most `seq_len`.

Other hyperparameters: preset defaults in `configs/models/BiMamba.toml`; tune generically.

## Differences

Clean-room implementation of the paper's patch, series-relation decider, Mamba+ gate, bidirectional encoder, and flatten-head equations; the author repository (no license file) was reference only and nothing was copied. Inputs are `[B, seq_len, enc_in]`, outputs `[B, pred_len, enc_in]`.
