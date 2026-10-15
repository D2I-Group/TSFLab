---
name: "DB2TransF"
description: "Attention-free inverted Transformer: variate (and calendar) tokens are mixed by multi-head, multi-level learnable Daubechies-2 wavelets. Use for multivariate data with cross-variable structure where attention is too costly; not for non-hourly calendar tokens or probabilistic output."
---

# DB2TransF

## Idea

- An inverted embedding maps each variate's normalized history (and, with `use_marks`, each hourly calendar feature) to one `d_model` token; non-affine `revin` wraps the model.
- `LearnableDB2` replaces attention: four learnable taps per feature, initialised from db2, compute approximation and detail coefficients along the token axis (Eqs. 10-11).
- `MultiLevelDB2` recurses `levels` times and sums the approximation and details interpolated back to the token count.
- `DB2Block` gives each head its own wavelet bank, then a residual projection and pre-norm GELU FFN; a linear head maps each variate token to the horizon.

## When to use

- Multivariate data where mixing across variate tokens helps but full attention is too costly.
- Calendar tokens help on hourly data with daily/weekly patterns; other rates need `use_marks = false`.
- Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count.
- `use_marks`: calendar tokens support hourly data only (`freq = "h"`); set `false` otherwise.

Other hyperparameters: preset defaults in `configs/models/DB2TransF.toml`; tune generically (`d_model` divisible by `n_heads`).

## Differences

- Independent rewrite after reading the pinned official code (no license); nothing copied.
- Multi-scale mixing interpolates and sums (code) instead of concatenating and trimming (paper Algorithm 1); high-pass taps use the code's sign.
- The normalization scale is detached; calendar tokens are optional and hourly-only. Detail in reference.md.
