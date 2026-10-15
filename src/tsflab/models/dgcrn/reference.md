# DGCRN — reference

## Paper

- **Title**: Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution
- **Venue**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 17, No. 1, Article 9
- **Published**: 2023 (arXiv: 2021-04)
- **arXiv**: https://arxiv.org/abs/2104.14917

## Abstract

Spatio-temporal traffic models ignore the dynamic correlations among road-network locations, RNN models are
inefficient, and fair comparison is lacking. DGCRN uses hyper-networks that generate dynamic filter parameters
at each time step from node attributes, filters node embeddings into a per-step dynamic graph integrated with
a predefined static graph, and limits decoder iterations during training for efficiency. The paper also opens a
standardized benchmark and a new traffic dataset; DGCRN outperforms 15 baselines on three datasets.

## Implementation mapping

- `DynamicGraphGenerator` is the hidden-state-conditioned hyper-network; `DynamicGraphConvolution` concatenates
  static forward/reverse and learned directed multi-hop propagation; `DynamicGraphGRUCell` inserts the filters
  into the recurrent gates.
- `adj_mx` is shape-checked and row-normalized in both directions; missing adjacency uses identity transitions.
- Historical and future raw or node-structured marks contribute one known time driver; future target values are
  never consumed.
- Designed from the method description and equations; the official source is reference only, and an earlier
  BasicTS-derived file was removed and was not a basis for this implementation.

## Citation

```bibtex
@article{DBLP:journals/tkdd/LiFYJYSJL23,
  author    = {Fuxian Li and Jie Feng and Huan Yan and Guangyin Jin and Fan Yang and Funing Sun and Depeng Jin and Yong Li},
  title     = {Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution},
  journal   = {{ACM} Trans. Knowl. Discov. Data},
  volume    = {17},
  number    = {1},
  pages     = {9:1--9:21},
  year      = {2023},
  url       = {https://doi.org/10.1145/3532611},
  doi       = {10.1145/3532611}
}
```
