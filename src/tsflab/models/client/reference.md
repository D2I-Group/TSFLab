# Client — reference

## Implementation details

- Independent rewrite of Section 3 and Algorithm 1. The MIT-licensed official repository
  (`daxin007/Client` at `802c603f`) was read in `models/Client.py`, `layers/RevIN.py`,
  `layers/Transformer_EncDec.py`, `layers/SelfAttention_Family.py` and the `scripts/` presets to
  resolve omissions; nothing was copied or imported.
- Shared components, each checked against the official layers: `revin` (affine, mean/std statistics
  detached, `eps=1e-5`, `eps^2` in the affine inversion), `self_attention_family.FullAttention` with
  `mask_flag=False` inside `AttentionLayer`, and `transformer_encdec` `EncoderLayer`/`Encoder`
  (post-norm, kernel-1 conv FFN, final LayerNorm).

## Differences in detail

- Token width is the lookback length: the official model overwrites `d_model` with `seq_len`, so the
  `d_model` values in its scripts have no effect.
- The linear module reads the RevIN-normalized input (Eq. 3), not the encoder output that Algorithm 1
  line 11 suggests.
- `w_lin` is a vector with one learnable entry per variate.
- Attention scales by `1/sqrt(d_head)` rather than the `1/sqrt(C)` written in Eq. 1.
- The channel count is fixed by `enc_in`; calendar marks and decoder inputs are ignored (the paper uses
  none); the unused `factor`, `output_attention` and non-forecast task options are omitted.
- The preset uses `d_ff=32` (official ETTh1 script) and `w_lin=1.0` (official default; the paper reports
  initial values between 0.5 and 1). Per-dataset searched hyperparameters are not reproduced, and
  reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{gao2023client,
  title   = {Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting},
  author  = {Gao, Jiaxin and Hu, Wenbo and Chen, Yuntian},
  journal = {arXiv preprint arXiv:2305.18838},
  year    = {2023}
}
```
