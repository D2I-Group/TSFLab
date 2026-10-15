# frwkv_linear_attention — reference

## Origin and granularity

Extracted with `frwkv` because FRWKV+ (`frwkvplus`) reuses FRWKV's frequency
encoder unchanged and only adds gating after it; neither model may import the
other. It follows the official code (`yangqingyuan-byte/FRWKV@d6dbf171`,
`model/FRWKV.py`, identical in `FRWKV-plus@08516fc0`), not the simplified
equations of the paper: "token shift" is an additive learned offset per
projection, `r`, `k`, `v` and the output map are two-layer GELU MLPs with a 1.2x
hidden width, the gate, replacement strength and decay come from rank
`max(64, d_model // 8)` low-rank MLPs, and the decay is confined to
`exp(-e^-0.3 sigmoid(tanh(.)))`. Embedding, FFT, gating and the forecast head
stay in the models. The encoder layers come from `transformer_encdec`.

## Invariants and equivalence evidence

- Equation and contract checks (at extraction) compared the scan against
  explicitly built transition matrices and confirmed that with unit decay and zero removal it
  equals causal (lower-triangular) linear attention, the first-step closed form
  `v (k^ . r)`, that the layer's output for a token does not depend on later
  tokens, the decay bounds, gradients, the residual identity of the branch with a
  zeroed output projection, and the `ValueError` cases.
- No stored numerical reference: the expected values were re-derived inside the
  checks. FRWKV and FRWKVPlus model checks exercised it end to end. All of these
  passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.

## Variants and options

Not covered: real token shift (interpolation with the previous token), the
paper's linear projections and Eq. 8 placement of the diagonal bonus matrix,
GroupNorm, chunked or parallel scans, multi-step state carry-over between
calls, and the full-attention alternative FRWKV+ offers for ablation.

## Related components

`transformer_encdec` (the encoder layers wrapped around it), `self_attention_family`
(softmax attention cores with the same call signature), `mamba` (another
recurrent selective-state mixer), `fft_extrapolation_conv` and `freq_band_moe`
(other frequency-domain operators).
