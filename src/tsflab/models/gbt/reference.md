# GBT — reference

## Differences in detail

- Implementation: independent rewrite from Sections 4.1-4.2 (Figs. 4-5, Eq. 3, Algorithm 1, Table 3) of the paper after reading the pinned official code (`OrigamiSL/GBT`, revision `4146c3865e7f09fe7f70e18b1477f676b509cccf`, Apache-2.0): `GBT/GBT.py`, `GBT/Auto_Regression.py`, `GBT/ConvBlock.py`, `Self_Regression/attn.py`, `Self_Regression/decoder.py`, `Self_Regression/embed.py`, `exp/exp_model.py`, `main.py`, `utils/tools.py`, and `scripts/`. The sinusoidal position table is the cataloged `embed.PositionalEmbedding` and the causal mask is `masking.TriangularCausalMask`.
- Resolved from the official code: value embeddings are kernel-1 convolutions with bias and Kaiming-normal weights (first-stage embeddings without positions); the first-stage self-attention uses weight-normalized point-wise Q/K/V/output projections with a residual add and no normalization; residual units use a stride-`s` WN convolution with kernel `kernel` then a kernel-3 WN convolution, dropout before each GELU, and a point-wise-conv / 3-2-1 max-pool shortcut; each pyramid branch has its own embedding and FC head over the flattened feature map.
- ESM: the scale comes from a point-wise projection of the decoder input per head and query, mapped by `3 ** (sigmoid(5 x) + 1e-5) - 1` (Algorithm 1 prints `3 sigma - 1`, which can be negative), and the Gaussian is evaluated at the key index.
- Decoder layers are post-norm with a `4 * d_model` GELU feed-forward and a final LayerNorm; the second-stage projection is added to the detached first-stage forecast; the attention context is read in the Informer `mix` layout (default on).
- The multivariate ETT/ECL/Traffic/weather/Exchange/ILI scripts run channel-independently (`--instance`); the scripts pass `--time`, which disables the calendar embedding, so no marks are used; the first stage reads the last `label_len` steps (equal to `seq_len` except for ILI, 18 of 36), exposed as `ar_len`.
- Defaults follow Table 3 (3 pyramid networks, 3 AR blocks, widths 32 / 512, kernel 3, dropout 0.1, learning rate 1e-4 halved per epoch, MSE, Adam) and `d_layers = 2` from the scripts.
- The official default extractor (`Attention`) fuses Res-P and Res-C into one stride-2 widening unit; Fig. 5a order is the default here, the fused form is `fused_conv = true`.
- `ar_epochs` defaults to 1 (the official `train_epochs` default), with no validation early stopping or checkpoint reload; the second stage does not reuse the stage-1 Adam state. Officially only the first-stage output is detached; its dropout still follows the module's train/eval mode, as in the official run. The official FC width assumes divisibility by the stride.
- The official ProbSparse attention option cannot accept the ESM argument, so it is not implemented.

## Verification

Checked properties: the ESM prior against Algorithm 1 and its monotone decay; scores add the prior and mask future keys; causality of the decoder attention; the `mix` context layout; the Res-P max-pool shortcut and length halving; AR block width doubling (paper and fused forms); the pyramid branch windows and averaging; `ar_len`; the forecast is the detached first stage plus the projected correction with no gradient into the first stage; channel independence; `pretrain` updates only the first stage and freezes it. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@article{shen2023gbt,
  title   = {GBT: Two-stage transformer framework for non-stationary time series forecasting},
  author  = {Shen, Li and Wei, Yuning and Wang, Yangzhu},
  journal = {Neural Networks},
  year    = {2023}
}
```
