---
name: "FRWKVPlus"
description: "FRWKV extended with cross-branch spectral gates (real and imaginary branches gate each other) corrected by trust-weighted periodic-position router context. Use for long-horizon multivariate forecasting with a known seasonal period and correlated channels; not for univariate or aperiodic series."
---

# FRWKVPlus

## Idea

- Backbone is FRWKV's: RevIN, learned scalar-to-vector embedding, orthonormal rFFT, one `FRWKVSpectralBranch` (component `frwkv_linear_attention`) each for the real and imaginary parts, one token per variable (Eqs. 1-5).
- Cross-branch gates: frequency-mean summaries `C_r`, `C_i` (Eq. 6) give base gates `G0_{i->r} = sigmoid(MLP(C_i))`, `G0_{r->i} = sigmoid(MLP(C_r))` (Eq. 7), so each branch scales the other.
- PPCE (Eqs. 8-12): the embedded window is extended to a multiple of `period_len` with its leading steps, averaged per within-period position, layer-normalized, and compressed by `num_routers` router queries into one context `C_pos` per variable.
- Trust-gated correction (Eqs. 13-17): `G = 1 + G0 + clip(alpha, 0, 0.2) * T * Delta`; correction heads start at zero and trust biases low, so training starts from the base gate.
- Gated spectra (gates broadcast over frequency) go through the inverse rFFT, are added to the embedding and mapped by FRWKV's MLP head; trains with the horizon-weighted L1 loss.

## When to use

- Designed for long-term multivariate forecasting where a known seasonal period shapes the spectrum (PPCE averages per within-period position).
- Mixes channels through linear attention over variable tokens; cost grows linearly with the channel count.
- Not for univariate series or data without a meaningful period; a wrong `period_len` turns the position context into noise.
- Like FRWKV, the recurrence is causal in column order, so permuting columns changes results.

## Configure

- `enc_in`: the dataset's channel count (number of variable tokens).
- `period_len`: the dominant seasonal period in steps (preset 24 for hourly data); the official recipes tune it per dataset and horizon. A period longer than the window is handled by cyclic padding.

Other hyperparameters: preset defaults in `configs/models/FRWKVPlus.toml`; tune generically.

## Differences

- Independent rewrite of Sec. 3 (Eqs. 1-22) of arXiv 2605.15690; the official release (AI Scientist Source Code License) was read for reference only.
- PPCE padding repeats the leading steps cyclically when the period exceeds the padded window (the official code fails there).
- The release's full-attention and other ablation modes (`encoder_attention_type`, `fre_flatten_mode`) are not provided.
- The catalog trainer replaces AdamW, the cosine schedule and the half-schedule early stopping; validation and test use the configured criterion while the weighted L1 drives training.
- Forward pass checked against a re-derivation of the official gated forward; reported benchmark numbers are not reproduction claims.

Full detail: `reference.md`.
