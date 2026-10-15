# KARMA — reference

## Paper

Ye et al., WASA 2025 (arXiv 2506.08939, https://arxiv.org/abs/2506.08939). Section 3 (Eqs. 2-12, 15, Fig. 1)
is the basis of this implementation.

## Implementation details from the official code

Inspected at `yedadasd/KARMA` revision `456d28f5405d90ef2b2a9c4dc32af24a3213d944`: `models/Karma.py`,
`layers/Embed.py`, `layers/RevIN.py`, `layers/Autoformer_EncDec.py`, `utils/cus_loss.py`,
`exp/exp_long_term_forecasting.py`, `run.py`, `scripts/ECL_script/Karma.sh`.

- The wavelet runs along the `d_model` axis of each embedded token (db4, one level, `symmetric` boundary mode of
  `pytorch_wavelets`), so the frequency Mambas have width `(d_model + 7) // 2` and scan across tokens like the
  temporal Mamba.
- Frequency coefficients are computed once from the embedded seasonal tokens and carried through the blocks
  without residuals.
- The reverse temporal branch applies its own RMSNorm, flips the token axis and flips the output back, and shares
  the temporal Mamba when the series is decomposed (separate weights otherwise).
- Encoder output is the inverse wavelet plus the temporal stream followed by LayerNorm; the trend's global Mamba
  has no residual or normalization; ATCD uses four heads and dropout `0.1`.
- Embeddings carry calendar tokens only without decomposition; RevIN is affine (the default `norm_method` falls
  back to it); every Mamba is `Mamba(d_model, d_state, d_conv, expand)`.
- The loss defaults to `htfLoss` = `0.2 * MSE + 0.8 * mean |rfft(y_hat) - rfft(y)|` along the horizon.
- Defaults `d_model = 512`, `e_layers = 2`, `d_state = 32`, `d_conv = 4`, `expand = 2`, `embed_dim = 128`,
  `moving_avg = 25`, `use_decomp = 1`, `dropout = 0.1` are the `run.py` defaults.

## Differences in detail

- The paper writes the first temporal input as `RMSNorm(X_se)` (Eq. 10) and the reverse branch without its own
  normalization or flip-back (Eq. 12); the official forms above are used.
- The db4 transform is the cataloged `orthogonal_dwt` (PyWavelets convention, half-sample symmetric extension,
  stride-2 analysis, transposed-convolution synthesis, filters not checkpointed), checked against `pywt`
  (symmetric-mode analysis coefficients, perfect reconstruction, coefficient length `(N + 7) // 2`); the cataloged
  `wavelet` component uses circular padding and has no db4 inverse, so it is not equivalent.
- The selective SSM is the cataloged pure-PyTorch scan (reference `dt` initialisation).
- The FAN/SAN normalization options, the unused attention-free variants, and the official per-dataset learning
  rates are not part of the model.
- Calendar tokens are built with `marks.adapt_tslib_marks`, which supports hourly `timeF` only.
- Checked: ATCD split (Eqs. 3-6), KarmaBlock updates (Eqs. 11-12) with the shared temporal Mamba, the inverse
  wavelet reconstruction (Eq. 15), decomposed and moving-average paths, the undecomposed path with calendar
  tokens, the Eq. (2) loss, and RevIN shift equivariance. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@inproceedings{ye2025karma,
  title     = {KARMA: A Multilevel Decomposition Hybrid Mamba Framework for Multivariate Long-Term Time Series Forecasting},
  author    = {Ye, Hang and Duan, Gaoxiang and Zeng, Haoran and Zhu, Yangxin and Meng, Lingxue and Zheng, Xiaoying and Zhu, Yongxin},
  booktitle = {International Conference on Wireless Artificial Intelligent Computing Systems and Applications (WASA)},
  year      = {2025}
}
```
