# TimeEmb — reference

## Paper

TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting (Xia et
al., NeurIPS 2025; arXiv:2510.00461).

Temporal non-stationarity makes forecasting unreliable, and existing methods conflate time-invariant
and time-varying components. TimeEmb separates (1) a time-invariant component, captured by a global
embedding module that learns persistent representations across series, from (2) a time-varying
component, processed by an efficient frequency-domain filter inspired by full-spectrum analysis. It
beats state-of-the-art baselines with fewer resources and can improve existing forecasters by simple
integration.

## Implementation notes

- Spectrum bank: real and imaginary tables `[slots, enc_in, seq_len // 2 + 1]`, initialized `N(0, 0.02)`.
- Dynamic filter: response `1 + 0j` plus `scale`-sized random imaginary part; the gate is `Linear -> GELU -> Linear -> Sigmoid` with hidden width `d_model` on the mean magnitude spectrum.
- Raw 6-field calendar marks give hour from field 4 and weekday from field 3; shorter marks use the last two fields.

## Citation

```bibtex
@misc{xia2025timeemb,
  author        = {Mingyuan Xia and Chunxu Zhang and Zijian Zhang and Hao Miao and Qidong Liu and Yuanshao Zhu and Bo Yang},
  title         = {TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.00461},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.00461}
}
```
