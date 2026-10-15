# APN — reference

## Paper

- **Title**: Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline
- **Venue**: AAAI 2026 (Oral) (arXiv 2505.11250, 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.11250

## Abstract

Irregular multivariate time series (healthcare, biomechanics, climate, astronomy) are hard to forecast because of irregularity and missingness, and existing methods are complex and resource-intensive.
APN's Time-Aware Patch Aggregation learns adjustable patch boundaries and a time-aware weighted average that turn irregular sequences into regularized representations channel-independently.
A simple query module integrates history and a shallow MLP predicts; APN beats prior methods in efficiency and accuracy on several real-world datasets.

## Citation

```bibtex
@inproceedings{liu2026apn,
  author    = {Xvyuan Liu and Xiangfei Qiu and Xingjian Wu and Zhengyu Li and Chenjuan Guo and Jilin Hu and Bin Yang},
  title     = {Rethinking Irregular Time Series Forecasting: A Simple Yet Effective Baseline},
  booktitle = {Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)},
  pages     = {23873--23881},
  year      = {2026},
  doi       = {10.1609/AAAI.V40I28.39563}
}
```
