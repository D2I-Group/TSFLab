# CoRA — reference

## Paper

- **Title**: CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2026-03)
- **arXiv**: https://arxiv.org/abs/2603.21828

## Abstract

Most time-series foundation models (TSFMs) are channel-independent and neglect channel correlations.
CoRA is a lightweight plug-and-play adapter, fine-tuned with a TSFM, that decomposes the correlation matrix
into low-rank time-varying (learnable polynomials over trends or periodic patterns) and time-invariant parts,
and learns positive and negative partial correlations through projection layers regulated by a
Heterogeneous-Partial contrastive loss at training time only. It improves TSFMs on 10 real-world datasets.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2603-21828,
  author    = {Hanyin Cheng and Xingjian Wu and Yang Shu and Zhongwen Rao and Lujia Pan and Bin Yang and Chenjuan Guo},
  title     = {CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter},
  journal   = {CoRR},
  volume    = {abs/2603.21828},
  year      = {2026},
  url       = {https://doi.org/10.48550/arXiv.2603.21828},
  doi       = {10.48550/ARXIV.2603.21828},
  eprinttype = {arXiv},
  eprint    = {2603.21828}
}
```
