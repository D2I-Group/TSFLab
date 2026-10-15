# TiDE — reference

## Paper

Long-term Forecasting with TiDE: Time-series Dense Encoder (Das, Kong, Leach, Mathur, Sen, Yu; TMLR
2023; arXiv:2304.08424).

Motivated by simple linear models outperforming Transformers in long-term forecasting, TiDE is an
MLP encoder-decoder that keeps the simplicity and speed of linear models while handling covariates
and nonlinear dependencies. Its simplest linear analogue provably attains near-optimal error for
linear dynamical systems under some assumptions; empirically it matches or beats prior methods on
long-term benchmarks while being 5-10x faster than the best Transformer.

## Implementation notes

- Normalization: per-channel lookback mean (detached) and standard deviation; the forecast is rescaled with the same statistics.
- `feature_projection` is a residual block from `time_feat_dim` to `feature_encode_dim` per step; the flattened projection over `seq_len + pred_len` steps is concatenated to each channel's lookback.
- Encoder: an input residual block plus `e_layers - 1` blocks of width `d_model`; decoder: `d_layers - 1` blocks then a dense block to `pred_len * decoder_output_dim`.
- Temporal decoder: a residual block `decoder_output_dim + feature_encode_dim -> d_ff -> 1` without LayerNorm (the paper's optional no-norm form).
- Future marks are cropped to the last `pred_len` steps of `x_mark_dec`.

## Citation

```bibtex
@article{DBLP:journals/tmlr/DasKLMSY23,
  author  = {Abhimanyu Das and Weihao Kong and Andrew Leach and Shaan Mathur and Rajat Sen and Rose Yu},
  title   = {Long-term Forecasting with TiDE: Time-series Dense Encoder},
  journal = {Trans. Mach. Learn. Res.},
  volume  = {2023},
  year    = {2023},
  url     = {https://arxiv.org/abs/2304.08424}
}
```
