# FAITH — reference

## Differences in detail

- Paper source: arXiv 2405.13300 v1 (Sections 3.1-3.3, Eqs. 1-19); the
  official repository links the journal version
  (doi:10.1016/j.knosys.2024.112790).
- Sources read: `models/MCTEM.py` (the FAITH model class),
  `models/local_global.py` (`series_decomp_multi`),
  `layers/Transformer_Encoder.py`, `layers/RevIN.py`, `run.py`,
  `scripts/FAITH_*.sh`. Nothing copied or imported.
- Resolved from the official code: moving-average kernels 17 and 49 with
  edge-replicate padding (the catalog `EdgePaddedMovingAverage` is the same
  operator), fusion weights initialised to `1/k`; the seasonal embedding
  multiplies every scalar by one learned `d_model` vector; FCEM selects bins from
  the first `(N + 1) // 2` channel-axis bins and FTEM from the first `L // 2`
  time-axis bins (a Nyquist bin of an even axis is always dropped); unselected
  bins are zeroed; one layer is shared by real and imaginary parts.
- The feature-extraction layer is a post-norm PatchTST-style encoder layer:
  multi-head scaled dot-product attention (reproduced by `self_attention_family`
  `AttentionLayer` + non-causal `FullAttention`), BatchNorm after attention and
  after a GELU feed-forward of width 256.
- Trend predictor: single `Linear(seq_len, pred_len)` with constant weights
  `1 / seq_len`; seasonal head flattens `(L, d)` with `head_dropout = 0.05`;
  fusion weights start at 1; RevIN is affine.
- Defaults follow the shipped scripts (`d_model = 32`, `n_heads = 4`,
  `e_layers = 1`, `modes = 64`). With `modes = 64`, `seq_len = 96` and at most
  128 channels every candidate bin is selected.
- The official layer would apply dropout in the projection and again on the
  residual. Training uses the catalog trainer and configured loss (official:
  MSE and Adam).
- Checked: the weighted multi-scale decomposition, bin pools and fixed random
  selection, FEM/iFEM identity reconstruction when all bins are kept, shared
  real/imaginary layer, FCEM and FTEM axes, seasonal embedding, trend
  initialisation, Eq. (4) fusion inside RevIN. Reported benchmark numbers are
  not reproduction claims of this implementation.

## Citation

Li, R., Jiang, M., Wang, K., Feng, K., Liu, Q., Sun, Y., Zhou, X. "FAITH: Frequency-domain Attention In Two Horizons for Time Series Forecasting." Knowledge-Based Systems 309, 112790 (2025). doi:10.1016/j.knosys.2024.112790.
