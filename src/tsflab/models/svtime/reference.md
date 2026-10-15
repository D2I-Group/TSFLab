# SVTime — reference

## Paper

SVTime: Small Time Series Forecasting Models Informed by "Physics" of Large Vision Model Forecasters
(Shen et al., arXiv:2510.09780, 2025).

Large pre-trained models are costly to train and serve. SVTime asks whether a compact model can match
large-model forecasting: it identifies key inductive biases of large vision-model (LVM) forecasters in
long-term forecasting and encodes them in crafted linear layers and constraint functions. Against 21
baselines on 8 benchmarks it beats lightweight state of the art and rivals large models with 10^3
fewer parameters than LVMs, with efficient training and inference in low-resource settings.

## Implementation notes

- `history_periods = seq_len // period`; only the last `history_periods * period` steps are used.
- `future_periods = ceil(pred_len / period)`; the forecast is cropped to `pred_len`.
- `PatchWisePeriodMap` holds one `[history_periods, history_periods + future_periods]` matrix per patch, so it emits backcast and forecast periods together.
- Trend: a linear map from the used history length to `future_periods * period`, applied to the backcast residual; the gate is a single scalar parameter.

## Citation

```bibtex
@misc{shen2025svtime,
  author        = {ChengAo Shen and Ziming Zhao and Hanghang Tong and Dongjin Song and Dongsheng Luo and Qingsong Wen and Jingchao Ni},
  title         = {SVTime: Small Time Series Forecasting Models Informed by "Physics" of Large Vision Model Forecasters},
  year          = {2025},
  eprint        = {2510.09780},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.09780}
}
```
