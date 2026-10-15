# STTN — reference

## Paper

Spatial-Temporal Transformer Networks for Traffic Flow Forecasting (Xu et al., arXiv:2001.02908, 2020).

Long-term traffic forecasting is hard because spatial-temporal dependencies are nonlinear and dynamic.
STTN introduces a spatial transformer, a graph-network variant that models directed spatial
dependencies with self-attention, where multiple heads capture different relations (similarity,
connectivity, covariance), and a temporal transformer for long-range bidirectional dependencies.
Composed into blocks, it trains fast and scales, with competitive results on PeMS-Bay and PeMSD7(M),
especially for long-term horizons.

## Implementation notes

- Four attention heads represent diverse directed spatial relations; the supplied graph contributes a separate stationary path whose output is fused with the attention output through a sigmoid gate.
- The positional parameter is learned (`seq_len x d_model`), not sinusoidal.
- The forecast layer flattens `seq_len * d_model` per node to `pred_len`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2001-02908,
  author  = {Mingxing Xu and Wenrui Dai and Chunmiao Liu and Xing Gao and Weiyao Lin and Guo{-}Jun Qi and Hongkai Xiong},
  title   = {Spatial-Temporal Transformer Networks for Traffic Flow Forecasting},
  journal = {CoRR},
  volume  = {abs/2001.02908},
  year    = {2020},
  url     = {http://arxiv.org/abs/2001.02908}
}
```
