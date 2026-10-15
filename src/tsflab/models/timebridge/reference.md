# TimeBridge — reference

## Paper

TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting (Liu et al., ICML 2025,
PMLR v267; arXiv:2410.04442).

Non-stationarity causes spurious regressions in short-term modelling but carries long-term
cointegration across variates. TimeBridge segments series into patches, applies Integrated Attention
to mitigate short-term non-stationarity and capture stable dependencies within each variate, and
Cointegrated Attention that preserves non-stationarity to model long-term cointegration across
variates. It reports state-of-the-art short- and long-term results and strong financial forecasting
on CSI 500 and S&P 500.

## Implementation notes

- Moving-average detrending supplies stationary queries and keys for intra-variate Integrated Attention while original patches remain the values.
- Attention downsampling aggregates long context into `long_count = max(1, num_p // stable_len)` tokens.
- Cointegrated Attention then mixes variates without removing non-stationarity.
- Patch embedding is `Linear(period, d_model)`; the head maps `long_count * d_model` to `pred_len` per variate.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/LiuWHL0BX25,
  author    = {Peiyuan Liu and Beiliang Wu and Yifan Hu and Naiqi Li and Tao Dai and Jigang Bao and Shu{-}Tao Xia},
  title     = {TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting},
  booktitle = {Forty-second International Conference on Machine Learning, {ICML} 2025},
  series    = {Proceedings of Machine Learning Research},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/liu25cb.html}
}
```
