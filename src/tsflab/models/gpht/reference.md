# GPHT — reference

## Differences in detail

- Implementation: local rewrite from Section 3 (Eqs. 3 to 6). The official code (`icantnamemyself/GPHT`, MIT) was inspected at revision `cfdbbd2f8386603476d0b48705b6189cfae0ad90` (`models/GPHT.py`, `exp/exp_long_term_forecasting_GPHT.py`, `layers/Embed.py`, `scripts/`) to resolve omissions; no source was copied.
- Resolved from the official code: each stage is `MaxPool1d(rate)`, a patch embedding with `patch = token_len / rate`, stride equal to the patch and no padding (bias-free linear value embedding plus sinusoidal positions, then dropout), a causally masked full-attention encoder with a final `LayerNorm`, and `Linear(d_model, token_len)` mapping every hidden state to the next token.
- Instance normalization is the biased variance plus `1e-5` and is recomputed on every rolled window; the next stage receives `x - PadFirstToken(out)` (right shift by one token).
- The training target is `x[token_len:]` followed by the first `token_len` steps of the label, scored with MSE on every position.
- The official model reads `GT_d_model`, `GT_d_ff`, `GT_e_layers` as single integers shared by all stages, as here, and takes `depth` separately from `GT_pooling_rate` (here `depth` is `len(pooling_rates)`).
- The official training script asserts `pred_len == token_len`; here the objective accepts any `pred_len >= token_len` and uses only the first `token_len` label steps, and `forward` rolls the window token by token for any `pred_len` (the official evaluation does this with a separate `ar_pred_len`).
- Training uses all channels and a matching target; the official script's `MS` slicing is not reproduced.
- Preset: `d_model=512`, `d_ff=2048`, `e_layers=3`, `n_heads=8`, `dropout=0.1` (the official pretraining defaults; its fine-tuning scripts use `dropout=0.0`); `seq_len` comes from the task (official default 336).
- Checked by structure and equation tests; no training was run.

## Citation

```bibtex
@inproceedings{liu2024gpht,
  title         = {Generative Pretrained Hierarchical Transformer for Time Series Forecasting},
  author        = {Liu, Zhiding and Yang, Jiqian and Cheng, Mingyue and Luo, Yucong and Li, Zhi},
  booktitle     = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year          = {2024},
  eprint        = {2402.16516},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2402.16516}
}
```
