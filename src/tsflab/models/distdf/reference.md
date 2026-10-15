# DistDF — reference

## Paper

- **Title**: DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.24574

## Abstract

Direct forecasting minimizes a conditional negative log-likelihood, usually estimated by MSE, which is biased
when the label sequence is autocorrelated. DistDF instead minimizes a joint-distribution Wasserstein discrepancy
between forecast and label sequences that provably upper-bounds the conditional discrepancy, is tractable and
differentiable, and improves diverse forecasting models.

## Citation

```bibtex
@misc{wang2025distdf,
  author    = {Hao Wang and Licheng Pan and Yuan Lu and Zhixuan Chu and Xiaoxi Li and Shuting He and Zhichao Chen and Haoxuan Li and Qingsong Wen and Zhouchen Lin},
  title     = {DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment},
  year      = {2025},
  eprint    = {2510.24574},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2510.24574}
}
```
