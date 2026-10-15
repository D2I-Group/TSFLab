---
name: "SWIFT"
description: "Lightweight channel-independent forecaster: RevIN, one-level Haar wavelet split, a shared conv across the two bands, a linear (or shallow MLP) sub-series map, inverse Haar. Use for long-term forecasting under tight compute or memory budgets; not for cross-channel structure, covariates, or probabilistic output."
---

# SWIFT

## Idea

- Each channel is normalized with `revin` and split by `HaarDWT1D` into approximation and detail bands of half length.
- A small shared Conv1d with a residual connection fuses the two bands (cross-band information fusion).
- `SubSeriesMapper`, a linear layer or shallow ELU MLP, maps half-length coefficients to half-horizon coefficients, shared across channels unless `not_independent=True` gives one per channel.
- `HaarIDWT1D` reconstructs the forecast and `revin` denormalizes it.

## When to use

- Long-term forecasting where a tiny linear-scale model is wanted (edge devices, efficiency studies); the mapper works on half-length sub-series.
- Channels that can be modelled independently; set `not_independent=True` only when channels differ strongly in dynamics.
- Not for data where cross-channel interactions, covariates, or calendar effects drive the target, or for probabilistic output.

## Configure

- `enc_in`: number of channels (sizes RevIN and per-channel mappers); input must be `[B, seq_len, enc_in]`.
- `seq_len` and `pred_len` must each be at least 2; odd lengths are replicate-padded by the Haar component.

Other hyperparameters: preset defaults in `configs/models/SWIFT.toml`; tune generically.

## Differences

- Independent rewrite from the paper; the official repository (inspected: `models/SWIFT_Linear.py`, `models/SWIFT_MLP.py`, `layers/RevIN.py` at `f1f4be7c4eeae09e75749b22090bc1b21cf33641`) is reference-only, no source copied (see THIRD_PARTY_NOTICES.md). Its README shows an MIT badge, but the tree has no LICENSE file, so the license is recorded as `NOASSERTION`.
- The official code uses the external `pytorch_wavelets` package (`DWT1DForward`/`DWT1DInverse`, Haar, `J=1`); the local `haar_dwt1d` component reimplements the orthonormal single-level Haar transform in closed form, lossless and algebraically identical for even lengths, and also supports odd lengths via a documented replicate-pad-and-truncate convention.
- The official `no_revin` flag (plain mean subtraction without rescaling) is dropped: RevIN (affine) is always used, matching the paper's default configuration.
