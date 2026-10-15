# ASTGCN — reference

## Paper

- **Title**: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
- **Venue**: AAAI 2019, pages 922-929

## Abstract

Traffic flows are highly nonlinear and existing methods do not model dynamic spatial-temporal correlations well.
ASTGCN has three independent components for recent, daily-periodic and weekly-periodic dependencies; each combines a spatial-temporal attention mechanism with a spatial-temporal convolution (graph convolution over space, standard convolution over time).
The three outputs are fused with learned weights; on two PeMS datasets it outperforms prior baselines.

## Formula mapping

Spatial and temporal attention is `SpatialTemporalAttention`; `AttentionChebyshevConvolution` implements the attention-modulated polynomial graph filter; `ASTGCNBlock` applies the gated temporal filter; `Model.forecast` maps history positions directly to the horizon. `adj_mx` is validated at construction.

## Citation

```bibtex
@inproceedings{guo2019astgcn,
  author    = {Shengnan Guo and Youfang Lin and Ning Feng and Chao Song and Huaiyu Wan},
  title     = {Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting},
  booktitle = {The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI 2019)},
  pages     = {922--929},
  year      = {2019},
  doi       = {10.1609/AAAI.V33I01.3301922}
}
```
