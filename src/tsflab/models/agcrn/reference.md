# AGCRN — reference

## Paper

- **Title**: Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
- **Venue**: NeurIPS 2020 (arXiv 2007.02842, 2020-07)
- **arXiv**: https://arxiv.org/abs/2007.02842

## Abstract

The paper argues that learning node-specific patterns is essential for traffic forecasting and that a pre-defined graph is avoidable.
It proposes a Node Adaptive Parameter Learning (NAPL) module to capture node-specific patterns and a Data Adaptive Graph Generation (DAGG) module that infers inter-dependencies among traffic series.
AGCRN combines both modules with recurrent networks to capture fine-grained spatial and temporal correlations, and outperforms prior work on two real-world traffic datasets without pre-defined spatial graphs.

## Citation

```bibtex
@inproceedings{bai2020agcrn,
  author    = {Lei Bai and Lina Yao and Can Li and Xianzhi Wang and Can Wang},
  title     = {Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting},
  booktitle = {Advances in Neural Information Processing Systems 33 (NeurIPS 2020)},
  year      = {2020},
  url       = {https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html}
}
```
