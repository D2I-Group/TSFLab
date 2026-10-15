# TimeBase — reference

## Paper

TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting (Huang, Zhou, Yang,
Yi, Wang, Wang; ICML 2025, PMLR v267).

Long-term forecasters usually rely on many parameters. Time series, unlike images or text, often show
temporal pattern similarity and low-rank structure, especially over long horizons. TimeBase, an
ultra-lightweight network, extracts core basis temporal components and turns point-level forecasting
into segment-level forecasting. It reaches competitive accuracy with remarkable efficiency and can
also serve as a plug-and-play complexity reducer for patch-based forecasters.

## Implementation notes

- Segments: `seg_num_x = ceil(seq_len / period_len)` input and `ceil(pred_len / period_len)` output segments; a ragged lookback is completed by repeating part of an earlier segment.
- `ts2basis`: linear `seg_num_x -> basis_num`; `basis2ts`: linear `basis_num -> seg_num_y`; both act per position within the period.
- Equations 1-4: segment `X`, `X_basis = BasisExtract(X_his)`, segment-level forecast map, flatten and trim to `pred_len`.
- Equations 5-7: `G = X_basis^T X_basis` and the off-diagonal Frobenius penalty, exposed as `aux_loss` in training.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/HuangZYYW025,
  author    = {Qihe Huang and Zhengyang Zhou and Kuo Yang and Zhongchao Yi and Xu Wang and Yang Wang},
  title     = {TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting},
  booktitle = {Forty-second International Conference on Machine Learning, {ICML} 2025},
  series    = {Proceedings of Machine Learning Research},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/huang25az.html}
}
```
