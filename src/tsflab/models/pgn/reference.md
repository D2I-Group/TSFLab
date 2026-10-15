# PGN — reference

## Differences in detail

- Implementation: independent rewrite from Sections 3.1 and 3.2 (Eqs. 1 to 5) of the paper. The official repository (`Water2sea/TPGN`, revision `4bbe2bee016094acd4299b2ad6fcd1db386bc99f`) has no license, so it is reference only: `models/TPGN.py` was read to resolve tensor layouts and omissions and nothing was copied or imported; its code, license and revision are therefore not recorded as source facts of this entry.
- Resolved from the reference code: the HIE kernel spans `rows - 1` earlier rows and excludes the current row; the gate and candidate come from one linear layer over `[value, calendar, H]` and are split in half; all linear maps are grouped per variate; the short-term branch feeds each row's columns through the row map and then a column map before repeating over the period; the head concatenates `[global, long]` (official order).
- Normalization: biased variance plus `1e-5` inside the square root (the same statistics as `revin` with `affine=False`); the official code lets the gradient flow through the standard deviation while `revin` detaches both statistics.
- Horizon alignment: `seq_len` and `pred_len` must be multiples of `period`. The official code rounds the number of forecast rows up and its experiment loop keeps the trailing `pred_len` outputs; that case is rejected here instead of misaligning the horizon.
- Calendar: the four hourly Time-Series-Library features built from the runner's raw marks via `marks.adapt_tslib_marks` (`freq="h"` only; other frequencies need preprocessed four-wide marks); absent marks are replaced by zeros. The official code also accepts minutely (`t`, five features) and daily (`d`, three features) calendars.
- Hyperparameters and variants: `d_model=64` and `period=24` are preset values, not the per-dataset searched hyperparameters of the paper; the TPGN-GRU/LSTM/MLP variants and the PGN-only (no short branch) result are limited to the `use_short_branch` switch. Reported benchmark numbers are not reproduction claims of this implementation.
- Structure checks covered: strictly causal HIE, gate equation, row independence, branch widths, gradients.

## Paper

PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting (NeurIPS 2024; arXiv 2409.17703, 2024-09).

## Citation

```bibtex
@inproceedings{jia2024pgn,
  title     = {PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting},
  author    = {Jia, Yuxin and Lin, Youfang and Yu, Jing and Wang, Shuo and Liu, Tianhao and Wan, Huaiyu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
