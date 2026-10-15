# REPNet — reference

## Differences in detail

Independent rewrite from arXiv 2507.05891 (v1, including Appendix C) after
reading RobertLeppich/REP-Net at `df3b2983` (no license file; nothing copied).
Defaults reproduce the released ETTh1-96 setup in `experiments/setups.py`:
extractors `[[3, 1, 1], [10, 2, 4], [15, 3, 5]]`, `encoding_size = 16`, `N = 3`,
`linear` representation, `posEmb` time embedding of size 16, no attention, GLU
on, per-variable feature mixing, no LSTM, `dropout = 0.3`, RevIN on. The ECL-96
setup corresponds to `representation = "cnn_3"`, `lstm_layers = 2`,
`time_embedding = ["timeF"]`, `time_embedding_size = 8` and four extractors.

- Attention, `tempEmb` and the time-embedding statistics follow the paper's
  stated intent rather than the buggy official code (see the issues in `card.toml`):
  attention runs across the patches of each variable, `tempEmb` embeds
  month/day/weekday/hour (and quarter-hour for `freq = "t"`) indices from raw marks,
  and time statistics are per sample.
- `timeF` features come from TSFLab's raw marks through `marks.adapt_tslib_marks`
  (hourly data can be rebuilt from raw marks; other frequencies need marks
  already in the Time-Series-Library `timeF` width). `posEmb` needs no marks.
- The time-window embedding follows the code layout: one Linear per extractor over
  the undilated `cover x time_embedding_size` window, value and time halves of
  `encoding_size / 2` each.
- The future-mark embedding computed by the official code is never used by the
  forecast and is omitted; `x_dec` and `x_mark_dec` are ignored.
- Training recipe (early stopping, scheduler, clipping, mixed precision) and the
  Huber loss belong to the run configuration and are not reproduced.
- Checked behaviour: patch extraction, the encoder variants (including the
  per-variable CNN path), time-informed patches, the positional table and
  per-sample time statistics, the memory-block residual structure, per-variable
  attention, the summed projection heads and sample independence. No checkpoint
  or published-metric reproduction is claimed.

## Citation

```bibtex
@article{leppich2025repnet,
  title   = {Decomposing the Time Series Forecasting Pipeline: A Modular Approach for Time Series Representation, Information Extraction, and Projection},
  author  = {Leppich, Robert and Stenger, Michael and Bauer, Andr{\'e} and Kounev, Samuel},
  journal = {arXiv preprint arXiv:2507.05891},
  year    = {2025}
}
```
