---
name: "FRWKV"
description: "Frequency-domain model: rFFT real and imaginary spectra each encoded by RWKV-7-style delta-rule linear attention over variable tokens. Use for long-horizon multivariate forecasting with correlated channels; not for univariate series or when results must not depend on column order."
---

# FRWKV

## Idea

- Lifts every RevIN-normalized value to an `E`-dimensional vector (one learned vector), applies an orthonormal rFFT along time (Eq. 1) and splits the spectrum into real and imaginary parts `[B, N, E, F]`.
- Each part goes through its own `FRWKVSpectralBranch` (component `frwkv_linear_attention`): `x + Linear(Encoder(Linear(x)))`, one token per variable (its flattened `E * F` spectrum), encoder layers inside post-norm `transformer_encdec` blocks.
- `FRWKVLinearAttention` is the RWKV-7-style recursion `S_t = (diag(w_t) - k~_t (a_t * k~_t)^T) S_{t-1} + v_t k^_t^T`, `y_t = S_t r_t` (Eqs. 5-7) with a bonus term and gated output (Eqs. 8-9), run sequentially over variable tokens in column order.
- The inverse rFFT (Eq. 2) is added back to the embedding; a three-layer GELU MLP maps each variable's flattened `L * E` representation to the horizon, then RevIN is inverted.
- Trains with the official horizon-weighted L1 `mean(|y_hat - y| (h + 1)^-0.5)` (`weighted_l1_loss`, `loss_alpha`).

## When to use

- Designed for long-term multivariate forecasting (ETT, ECL, Exchange, Weather, Solar in the paper) where cross-variable spectral structure helps.
- Mixes channels: the recurrence runs over variable tokens, so its cost grows linearly with the channel count.
- Not for univariate series (a single token leaves the attention with nothing to mix).
- The recurrence is causal in column order: a variable's forecast depends only on itself and earlier columns, so permuting columns changes results.

## Configure

- `enc_in`: the dataset's channel count; it also fixes the number of variable tokens the recurrence runs over.
- Any `seq_len >= 2` works (`seq_len // 2 + 1` frequency bins); no divisibility constraint.

Other hyperparameters: preset defaults in `configs/models/FRWKV.toml`; tune generically.

## Differences

- Independent rewrite of Sec. 2 (Eqs. 1-9) of arXiv 2512.07539 v2; the official repository has no license (`NOASSERTION`) and was read for reference only. Where paper and code disagree the code is followed (see `issues` in `card.toml`).
- One token per variable (not per time step); additive learned offsets instead of token shift; MLP r/k/v/output maps; RWKV-7 removal term; the code's bonus form; no decoder (MLP head).
- Linear attention re-expressed as a batched loop over tokens with identical algebra; unused official parameters are not created.
- The catalog trainer replaces the official schedule (`lradj type1`, early stopping); validation and test use the configured criterion while the weighted L1 drives training.
- Reported benchmark numbers are not reproduction claims of this implementation.

Full detail: `reference.md`.
