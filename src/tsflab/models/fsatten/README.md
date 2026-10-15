---
name: "FSatten"
description: "Variate Transformer whose query/key are per-head scaled FFT amplitude spectra (MSS) and whose values come from whole-series tokens. Use for multivariate forecasting where channels relate through shared spectral content; not for univariate series or calendar-driven tasks."
---

# FSatten

## Idea

- `amplitude_spectrum` (Eqs. 3-4): rFFT over the lookback of every normalized variate and its modulus, giving `F = seq_len // 2 + 1` phase-free amplitudes per variate.
- `MultiHeadSpectrumScaling` (MSS, Eq. 6): `MSS(A) = A o W_h` with one learnable `[variates, F]` matrix per head, a Hadamard product so frequency axes stay orthogonal.
- `FrequencySpectrumAttention` (Eq. 5): `Q = MSS_Q(A)`, `K = MSS_K(A)`, scaled dot product over variate tokens with `1/sqrt(F)`; `V = Linear_V(Emb(X))` from the latent tokens, then an output projection.
- `Model` is the paper's Variate Transformer (iTransformer layout): instance normalization, one whole-series token per variate, post-norm encoder layers with FSatten, a final LayerNorm, and a linear map to the horizon.

## When to use

- Designed for multivariate forecasting where the relation between channels is visible in their amplitude spectra (shared periodicities); attention weights are phase-free and depend only on the input spectrum.
- Mixes channels by attention over variate tokens; cost is quadratic in the channel count.
- Not for univariate series (one token, nothing to attend to) or tasks that need calendar marks (marks are ignored).

## Configure

- `enc_in`: the dataset's channel count; every variate has its own MSS weight row, so a trained model is tied to this count and column order.
- `seq_len` sets `F = seq_len // 2 + 1`, the query/key width; no divisibility constraint.

Other hyperparameters: preset defaults in `configs/models/FSatten.toml`; tune generically.

## Differences

- Independent implementation from Sections 2.2, 3.1-3.2 (Eqs. 2-6) and Appendix B.2. The official repository `Joeland4/FSatten-SOatten` (revision `c1b0b61`) holds only a README ("will be released soon") and training logs, so no code, revision or license is recorded.
- From the official logs (argument namespaces only): `d_f = 49 = 96 // 2 + 1` frequencies for `seq_len = 96`, iTransformer-style arguments (`use_norm`, `class_strategy`), and `d_model = d_f * n_heads` (e.g. 490 with 10 heads on ECL). Here `d_model` is free; the per-head value width is `d_model / n_heads`, the query/key width always `F`.
- Local choices where the paper is silent: one amplitude spectrum of the normalized input feeds every layer's query/key scaling (per-layer MSS weights, passed through the shared encoder layer's `delta`); MSS weights start at one; unnormalized rFFT as in Eq. (3) (`fft_norm="backward"`; `ortho`/`forward` are options); dropout on the attention weights; no calendar-mark tokens.
- Not implemented: SOatten (learnable orthogonal embedding with Head-Coupling Convolution), the paper's separate attention. Reported benchmark numbers are not reproduction claims.
- Checked: DFT modulus, phase invariance, per-head Hadamard scaling, Eq. 5/6 attention weights, Q/K independent of tokens, value path, normalization reversal, gradients.
- Citation: Wu, H. "Revisiting Attention for Multivariate Time Series Forecasting." arXiv:2407.13806, 2024.
