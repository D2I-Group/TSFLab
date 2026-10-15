# FRWKV — reference

## Differences in detail

- Implementation: independent rewrite of Sec. 2 (Eqs. 1-9) of arXiv 2512.07539 v2. The official repository (`yangqingyuan-byte/FRWKV`, revision `d6dbf171b78864bdba93ee692152b3a5918b4852`) has no license file, so it is recorded as `NOASSERTION` and used as reference only: `model/FRWKV.py`, `layers/Transformer_EncDec.py`, `layers/RevIN.py`, `experiments/exp_forecast.py`, `run.py` and `scripts/FRWKV/*.sh` were read to resolve omissions; nothing was copied or imported.
- Resolved from the official code: affine RevIN; a learned `[E]` embedding vector initialized `N(0, 0.1^2)` (no embedding when `E = 1`); orthonormal rFFT with `seq_len // 2 + 1` bins; per part `Linear(E * F, d_model)`, `e_layers` post-norm encoder layers (conv feed-forward of width `d_ff`, GELU) with a final LayerNorm, `Linear(d_model, E * F)` and a residual; the linear-attention layer of the `frwkv_linear_attention` card; inverse rFFT added to the embedding; head `Linear(L * E, d_ff)`, GELU, `Linear(d_ff, d_ff // 2)`, GELU, `Linear(d_ff // 2, H)`.
- Internal dropout fractions follow the code: encoder layers use half of `dropout`, the attention output a quarter, the head dropouts 0.3, 0.2 and 0.5 of it.
- Preset settings come from the ETTh1 script: `d_model = d_ff = 512`, 8 heads, `embed_size = 16`, 2 layers, dropout 0.2, learning rate 1e-4, batch 32 (3 layers and dropout 0 for ECL and Weather).
- Shared components, checked against the reference: `revin.RevIN(affine=True)` matches the official `RevIN` (detached mean and `sqrt(var + 1e-5)`, affine inverse with `eps^2`); `transformer_encdec.Encoder` / `EncoderLayer` match the official `Encoder_ori` / `EncoderLayer` (post-norm residuals, kernel-1 convolutions, final LayerNorm; the official CKA logging is omitted); `frwkv_linear_attention` was extracted from this model and is shared with FRWKVPlus.
- Training: the official schedule, `lradj type1`, and early stopping settings are not reproduced.

## Verification

- The forward pass matches an einsum loop re-derivation of the official model. Checked properties: the frequency branches receive one `E * F` token per variable; a variable's forecast depends only on itself and earlier columns; RevIN affine equivariance; the official dropout fractions; gradients; the horizon-weighted L1 objective and its switch; the strict schema.
- The shared recursion was checked by the `frwkv_linear_attention` component check at extraction; it passed in the full suite run of 2026-10-03 before the test suite was consolidated.

## Citation

```bibtex
@article{yang2025frwkv,
  title   = {FRWKV: Frequency-Domain Linear Attention for Long-Term Time Series Forecasting},
  author  = {Yang, Qingyuan and Deng, Shizhuo and Chen, Dongyue and Teng, Da and Gan, Zehua},
  journal = {arXiv preprint arXiv:2512.07539},
  year    = {2025}
}
```
