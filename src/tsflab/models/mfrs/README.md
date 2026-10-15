---
name: "MFRS"
description: "Channel-independent cross-attention from variate tokens to synthetic reference waveforms built from periods and harmonics extracted from the training spectrum. Use for strongly periodic, multi-seasonal data with timestamps, including many channels; not for aperiodic or noise-like series."
---

# MFRS

## Idea

- `primary_base_patterns` (Algorithm 1) re-indexes the training amplitude spectrum by period (`Psi(T) = Phi(1/T)`, each rFFT bin assigned to its nearest integer period) and keeps `T` when `Psi(T)` is the largest of `Psi(1..2T)`.
- `harmonic_base_patterns` (Algorithm 2) scores harmonics `k / T_m` by `sum_c Phi_c(k / T_m) / Phi_c(1 / T_m)` and keeps the top `harmonics` (`Q`); `periods` adds manual base patterns.
- `reference_waveform` (App. B) builds one series per pattern `(T, k)` from `phase = (t k) mod T` (sine, sawtooth, rectangle or pulse); `t` comes from the window's timestamps, or from Algorithm 3 (best Pearson alignment against a stored training stretch) when dates are absent.
- `ReferenceCrossAttentionLayer`: standardized variate tokens (queries) attend only to reference-series tokens (keys and values) through a shared `Linear(S -> d)` embedding; a LayerNorm and `Linear(d -> H)` give the forecast, de-standardized. Variates never attend to each other (channel independent).

## When to use

- Designed for data whose structure is dominated by several periods and their harmonics (multi-frequency seasonality).
- Phases are anchored to absolute time, so timestamps matter (Algorithm 3 is a fallback).
- Channel independence keeps cost linear in channels, targeting scalable forecasting on many-channel data.
- Not for aperiodic, trend-driven, or noise-like series: without spectral peaks the reference set is empty or meaningless.

## Configure

- `enc_in`: number of variates.
- `max_period`: upper bound of the period search; must exceed the longest seasonal period in steps (default 1000).
- `periods`: optional manual base periods in steps (integers >= 2), e.g. the known daily or weekly period.
- `extract`: base patterns come from the training-split spectrum (`training_setup` before the first epoch); before fitting the reference set is the manual `periods` or one period equal to `seq_len`.

Other hyperparameters: preset defaults in `configs/models/MFRS.toml`; tune generically.

## Differences

- Independent implementation from arXiv 2503.08328v1; the official repository is unavailable (404), so there is no reference comparison. Fidelity is `inferred`.
- Width, depth, heads, dropout, optimizer and loss are not reported; inverted-Transformer defaults are used (`d_model = d_ff = 512`, 8 heads, 2 layers).
- `Lp` and `Q` are not reported: `max_period = 1000`, `harmonics = 8`, no manual periods.
- Paper errors fixed: `Psi(1) = 0`; Algorithm 2 periods ascend and harmonics are inserted once.
- The reference-series length equals the look-back window; reference series are standardized and embedded once as fixed keys and values.
- Further resolutions are in reference.md and the card's `issues`.
