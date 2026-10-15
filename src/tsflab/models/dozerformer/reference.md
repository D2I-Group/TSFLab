# Dozerformer — reference

## Differences in detail

Defaults follow `scripts/ETTh1.sh` (`patch_size = 24`, `local_window = 3`,
`stride = 7`, `vary_len = 1`, `embed_dim = 8`, `label_len = 96`) and `run.py`
(`n_heads = 4`, 2 encoder and 1 decoder layers, `d_ff = 32`, `dropout = 0.2`).

- The official loader takes the decoder history from `batch_y[:, :label_len]`,
  which starts `label_len` steps before the end of the history; TSFLab takes the
  same values from `x_enc`.
- The official per-query loop that fills selected scores is replaced by a dense
  score matrix multiplied by a boolean selection (`score_mode = "zero"`); values
  and gradients are the same, and the claimed compute saving is not realised (as
  in the official code). Unselected pairs keep a pre-softmax score of 0.
- `local_window = 0` or `stride = 0` disables that component as in the code,
  while Vary is always active in cross-attention as in the code. Cross Stride is
  anchored at encoder patch `N_enc - floor(N_dec/2)`.
- Encoder and decoder layers are post-norm with a depthwise-conv (kernel 3)
  feed-forward, wrapped by pre/post LayerNorm, learnable per-variable position
  tables and a skip from the patch embedding, as in the official code.
- One embedding width serves encoder and decoder (`run.py` forces the official
  `decoder_embed_dim` to equal `embed_dim`).
- The self and cross masks were checked against hand-derived matrices, as were
  the zero/masked score semantics, the
  decomposition, decoder input, patch-token layout and the seasonal-plus-trend
  head.

## Citation

Zhang, Y., Wu, R., Dascalu, S. M., Harris, F. C., Jr. "Sparse Transformer with Local and Seasonal Adaptation for Multivariate Time Series Forecasting." Scientific Reports 14, 15909 (2024).
