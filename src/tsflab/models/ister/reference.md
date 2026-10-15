# Ister — reference

## Differences in detail

- The preset follows `scripts/CD_Ister/ETT_script/Ister_ETTh1.sh` at horizon 96 (`layers = 2`, `d_model = 128`,
  `d_ff = 2048` from the `run.py` default, `moving_avg = 25`, `label_len = 48` as `recon_len`); other datasets
  and horizons in the scripts change `layers`, `d_model`, and `d_ff`.
- Calendar features: TSFLab's raw six-column marks are converted with `adapt_tslib_marks` to the TSLib hourly
  `timeF` layout; only `freq = "h"` can be reconstructed from raw marks.
- The reconstruction part of the training loss uses the last `recon_len` steps of the history window, which is
  what the official `batch_y[:, :label_len]` contains; validation and test metrics use the horizon only.
- The output projection (a fourth matrix beyond the paper's `W_Q, W_K, W_V`) is kept as in the code.
- Equation (1) with and without the official GELU, the official forward order, the head initialization, the
  calendar-token handling, and the training objective were checked against the official code at the pinned
  revision.

## Citation

```bibtex
@inproceedings{cao2026ister,
  title     = {Ister: Linear Transformer for Efficient Multivariate Time Series Forecasting},
  author    = {Cao, Fanpu and Yang, Shu and Chen, Zhengjian and Liu, Ye and Cui, Laizhong},
  booktitle = {ICASSP 2026 - IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages     = {3571--3575},
  year      = {2026},
  doi       = {10.1109/ICASSP55912.2026.11463971}
}
```
