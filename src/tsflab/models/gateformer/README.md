---
name: "Gateformer"
description: "Transformer that gate-fuses a global-window and a patch-attention embedding per variate, then applies cross-variate attention with a second gate. Use for multivariate forecasting where both within-series temporal structure and cross-channel dependence matter; not for univariate series."
---

# Gateformer

## Idea

- Combines cross-time and cross-variate attention through learned gates instead of concatenation or a fixed order (paper Sec. 3, Fig. 2).
- Each variate gets a global embedding (the whole window projected to `d_model`, as in the inverted embedding) and a temporal embedding (patches encoded by a shared Transformer across patches, flattened by `flatten_forecast_head`).
- `gate_global_temporal` (`gated_fusion`) blends the two embeddings per variate.
- A variate-wise Transformer encoder attends across variates, and `gate_variate` blends its output with its input.
- A linear layer projects each variate token to `pred_len`; the mean/std standardization is restored on the output.

## When to use

- Designed for multivariate forecasting where both temporal patterns inside each series and dependencies between variates matter; the gates let training weigh the two.
- Mixes channels by attention over variate tokens; cost is quadratic in the channel count.
- Not for univariate series (cross-variate attention has nothing to mix). Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count (number of variate tokens).
- `patch_len`, `stride`: `patch_len` must not exceed `seq_len`; the window is end-padded by `stride` steps, giving `(seq_len + stride - patch_len) // stride + 1` patches.

Other hyperparameters: preset defaults in `configs/models/Gateformer.toml`; tune generically.

## Differences

- The official implementation (`models/Gateformer.py`, `layers/Embed.py`, revision `298fd10`) was inspected to resolve patch padding (`stride`-length end replication) and the exact gate equations. It has no `LICENSE` file, so `code` provenance is omitted and nothing was copied or adapted.
- The official code uses a raw sinusoidal position table; the reused `positional_encoding` component standardizes it (zero mean, unit-scaled variance), a minor magnitude difference with the same relative-position content.
