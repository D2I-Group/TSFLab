# Autoformer — reference

## Paper

- **Title**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
- **Venue**: NeurIPS 2021 (arXiv 2106.13008, 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.13008

## Abstract

Intricate long-term temporal patterns prevent self-attention from finding reliable dependencies, and sparse point-wise attention creates an information bottleneck.
Autoformer renovates series decomposition as an inner block of the network, giving progressive decomposition, and introduces Auto-Correlation, based on series periodicity, which discovers dependencies and aggregates representations at the sub-series level.
Auto-Correlation beats self-attention in efficiency and accuracy; Autoformer reports a 38% relative improvement on six long-term benchmarks (energy, traffic, economics, weather, disease).

## Citation

```bibtex
@inproceedings{wu2021autoformer,
  author    = {Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  title     = {Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS 2021)},
  pages     = {22419--22430},
  year      = {2021},
  url       = {https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html}
}
```
