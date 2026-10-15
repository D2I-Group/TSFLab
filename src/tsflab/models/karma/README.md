---
name: "KARMA"
description: "Mamba hybrid: attention-based seasonal-trend split, db4 wavelet plus bidirectional Mamba blocks over seasonal variate tokens, global Mamba on trend tokens. Use for multivariate long-term forecasting with trend and seasonal structure; not for non-hourly data with calendar tokens."
---

# KARMA

## Idea

- ATCD (Sec. 3.3): channel projection, self-attention over time with SiLU gives the trend, the remainder is seasonal (Eqs. 3-6); `decomposition = "moving_avg"` swaps in `series_decomposition`, which the official code runs.
- Each component is embedded as one token per variate (`DataEmbedding_inverted`); trend tokens pass through one global Mamba.
- HFTD (Sec. 3.4): `orthogonal_dwt` (db4, `symmetric`) splits each seasonal token along its embedding axis into low/high-frequency coefficients; `KarmaBlock`s update them with separate Mambas and the tokens with a bidirectional Mamba across variates (Eqs. 11-12), and the inverse wavelet plus the temporal stream rebuild the seasonal representation (Eq. 15).
- Seasonal and trend representations are summed, projected `d_model -> pred_len` per variate, and denormalized by affine RevIN.
- `hybrid_loss` (Eq. 2): `time_loss_weight` times the time-domain loss plus the rest times the mean modulus of the horizon rfft residual.

## When to use

- Multivariate long-term forecasting where series carry both trend and seasonal components, with mixing across variate tokens.
- Mamba scans keep cost linear in the number of variate tokens.
- Calendar tokens exist only without decomposition and only for hourly data.
- Heavy model (default `d_model = 512`); a poor match for very short training data.

## Configure

- `enc_in`: number of data channels (variate tokens).
- `moving_avg`: odd moving-average window, used only with `decomposition = "moving_avg"`.
- `freq` / `use_marks`: only `h` is supported; set `use_marks = false` for other sampling rates.

Other hyperparameters: preset defaults in `configs/models/KARMA.toml`; tune generically.

## Differences

- Independent rewrite from Sec. 3 after reading the pinned official code; nothing copied (repository has no license file, recorded `NOASSERTION`).
- ATCD by default; the official code hard-codes `karma_decomp = False` and never runs it (`decomposition = "moving_avg"` reproduces that path).
- Official forms of Eqs. (10)/(12) (own RMSNorm and flip-back on the reverse branch, shared temporal Mamba).
- The unused official `TaylorKANLayer` is not reproduced.
- Pure-PyTorch selective scan instead of `mamba_ssm` fused kernels; results may differ slightly.
- Eq. (2) weighted as in the official code (`time_loss_weight = 0.2`).
