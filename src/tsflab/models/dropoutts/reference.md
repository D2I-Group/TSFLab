# DropoutTS — reference

## Differences in detail

- Sources checked: the paper (Sec. 3-4, Eq. 1-11, implementation details and
  hyperparameter study) and, at the pinned revision,
  `src/basicts/modules/dropout_ts.py`,
  `src/basicts/runners/callback/dynamic_dropout.py`, `run_baselines.py` and the
  BasicTS PatchTST config. Output is the `[B, pred_len, enc_in]` point forecast.
- Backbone: the paper evaluates Informer, Crossformer, PatchTST, iTransformer,
  TimeMixer and TimesNet (BasicTS). This entry uses the catalog `patchtst`
  component (the default model of `run_baselines.py`), sized after BasicTS
  PatchTST (1 layer, width 256, 1 head, FFN 1024, affine RevIN). Its encoder is
  `nn.TransformerEncoderLayer` with a final BatchNorm.
- Dropout sites: as officially, every `nn.Dropout` is replaced, including those
  with configured rate 0, so in training all use the adaptive rate; `dropout`
  applies only when a replaced module is called without rates. Rates of length
  `B` are repeated over channel-folded `[B * C, ...]` tensors.
- SFM: the official code computes flatness on `(log(1 + A))^2` with `1e-8`
  floors; scale and bias are `[1, C, 1]`, initialized to 1 and 0.
- Defaults: `p_min = 0.05`, `p_max = 0.5`, `init_alpha = 10`,
  `init_sensitivity = 1.0` (paper Sec. 5.5 and `run_baselines.py`); the code's
  class defaults are `p_min = 0.1`, `init_sensitivity = 5.0`.
- Optimization: scorer parameters train with the backbone's optimizer and
  learning rate (the official callback adds them as a parameter group with the
  same rate). The official layout check (`shape[1] != shape[2]`) is replaced by
  the fixed `[B, L, C]` contract.
- Data: official experiments use BasicTS scaling, Adam, 100 epochs with early
  stopping and clean-target evaluation on synthetic data; the repository's
  trainer and scaling apply here.
- Venue: the pinned arXiv version (v2, May 2026) carries the ICML 2026
  proceedings footer.

## Citation

Zhong, S., Liu, Y., Cui, Z., Shao, Z., Wang, F., Wen, Q., Liang, Y. "DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting." ICML 2026. arXiv:2601.21726.
