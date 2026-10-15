# D2STGNN — reference

## Paper

- **Title**: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting
- **Venue**: VLDB 2022
- **Published**: 2022 (arXiv: 2022-06)
- **arXiv**: https://arxiv.org/abs/2206.09112

## Abstract

Traffic data from road-network sensors contains two hidden signals: diffusion signals and inherent signals,
yet prior spatial-temporal GNNs treat traffic entirely as diffusion. The Decoupled Spatial-Temporal Framework
(DSTF) separates them in a data-driven way with an estimation gate and residual decomposition, and
D2STGNN instantiates it with a dynamic graph learning module, advancing the state of the art on four
real-world traffic datasets.

## Citation

```bibtex
@article{DBLP:journals/pvldb/ShaoZWWXCJ22,
  author    = {Zezhi Shao and Zhao Zhang and Wei Wei and Fei Wang and Yongjun Xu and Xin Cao and Christian S. Jensen},
  title     = {Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {15},
  number    = {11},
  pages     = {2733--2746},
  year      = {2022},
  url       = {https://www.vldb.org/pvldb/vol15/p2733-shao.pdf},
  doi       = {10.14778/3551793.3551827}
}
```
