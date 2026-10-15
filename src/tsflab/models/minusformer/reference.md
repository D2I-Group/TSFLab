# Minusformer — reference

## Paper

Liang, Zhang, Yuan, Zhang, arXiv 2402.02332 (2024-02, revised 2024-06). Sources used: Section 2.4 (Eqs. 2 and 5-9,
Fig. 4) and Appendix H (Algorithm 1).

## Differences in detail

**Official code read** (`Anoise/Minusformer` at `78b3031c`): `model/Minusformer.py`, `layers/Transformer_EncDec.py`,
`layers/SelfAttention_Family.py`, `utils/tools.py`, `run.py`, and the `scripts/` presets.

**Resolved from the official code.** The embedding is a single linear map over the time axis without dropout or
positional terms, with calendar features appended as extra tokens and dropped before the output. The attention is
the TSLib full attention (`AttentionLayer` with Q/K/V/output projections, attention dropout equal to `dropout`),
whose output is subtracted after dropout. The feed-forward network is two pointwise linear maps
(`d_model -> d_ff -> d_model`) with dropout after the activation and after the second map; the activation is GELU
with attention and ReLU without. The output-stream gate reads the concatenation of the raw attention output and
the feed-forward output (only the feed-forward output when `attn = false`). The Dirac switch `delta` of Eq. (7) is
the `attn` flag; `d_block` defaults to the horizon. The standardization keeps gradients through the mean and scale.
Preset sizes follow the official `run.py` defaults (`d_model = 512`, `n_heads = 8`, `e_layers = 2`,
`d_ff = 2048`, `dropout = 0.1`, `attn = gate = true`); several ETT scripts set `attn = 0` and per-horizon `e_layers`.

**Other.** Calendar tokens are built from raw marks with `marks.adapt_tslib_marks` (hourly `timeF` only).
Learning-rate schedules are run settings, not model behaviour.

**Verified properties.** The alternating-subtraction output stream against Eq. (2); the block's input-stream
subtraction and gate (Eqs. 5, 7, 8) and gated output stream (Eq. 9) with and without attention and with the gates
off; the scaler round trip and the forecast's scale/shift equivariance; calendar-token handling; the `d_block`
alignment; the forward composition; gradient flow; the schema. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@article{liang2024minusformer,
  title   = {Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals},
  author  = {Liang, Daojun and Zhang, Haixia and Yuan, Dongfeng and Zhang, Bingzheng},
  journal = {arXiv preprint arXiv:2402.02332},
  year    = {2024}
}
```
