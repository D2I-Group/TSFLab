---
name: "ASGMamba"
description: "Patch Mamba at scales 8/16/32 whose tokens are gated by patch rFFT low/mid/high band energies; softmax scale fusion. Use for long-term multivariate forecasting with mixed-frequency content; not for graph-structured or probabilistic tasks."
---

# ASGMamba

## Idea

- `band_energy` (Sec. 4.2.1, Eq. 4): patch-level rFFT power summed into low (`nu <= 1/3`), mid, and high (`nu > 2/3`) bands of the normalized frequency `nu_j = 2j / P`, divided by the patch's total energy.
- `SpectralGate` maps that descriptor to a gate over each patch token's features: the pinned official `1 + 0.5 tanh(W2 SiLU(W1 v))` (zero-initialised, starts at one) or the paper's `sigmoid(W2 ReLU(W1 v))` (Eq. 5).
- `ScaleBranch` (Fig. 1): zero-padded patching, `PatchEmb + E_pos + E_node` (Eq. 3), `Z_out = Dropout(Mamba(LayerNorm(Z) * G)) + Z` (Eq. 6) with the shared `MambaBlock`, and a flatten head.
- Branch forecasts are fused by a learnable global softmax over scales (Eq. 8), inside RevIN.

## When to use

- Long-term forecasting where local frequency content varies over time (smooth vs. noisy segments) and several patch scales are informative.
- Channels share weights but carry a learned node embedding; no explicit cross-channel attention.
- Linear-time Mamba scan per scale; point output only.

## Configure

- `enc_in`: number of channels.
- `patch_sizes` are right-zero-padded to fit `seq_len`, so there is no divisibility constraint.

Other hyperparameters: preset defaults in `configs/models/ASGMamba.toml`; tune generically.

## Differences

Paper: Li, Zhang, Wang, Wei, Xing, *ASGMamba: Adaptive Spectral Gating Mamba for Multivariate Time Series Forecasting*, arXiv 2602.01668 (2026). Independent rewrite after reading the pinned official code (its Apache-2.0 text is inherited from the DLinear template); nothing copied.

- Patches are non-overlapping and right-zero-padded as in the code; `overlap = true` gives the paper's stride `P/2`.
- The gate defaults to the code's `identity_centered` form in `(0.5, 1.5)`; `gate = "sigmoid"` gives Eq. (5).
- The residual adds the ungated, un-normalized input after dropout (Sec. 4.2.2 and code), not Algorithm 1 line 14.
- Band edges follow the paper's normalized frequency (1/3, 2/3 of Nyquist).
- The pure-PyTorch shared `MambaBlock` is always used; the scripts' unused `--e_layers`/`--d_ff` are not exposed.
- Unstated details (gate width `max(D/4, 8)`, `N(0, 0.02^2)` embeddings, ones scale logits, flatten head, dropout on the Mamba output only, affine RevIN) come from the code.
