# TimeFilter — reference

## Paper

TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting (Hu et al.,
ICML 2025, PMLR v267; arXiv:2501.13041).

Channel-independent forecasting overlooks covariate relations; channel-dependent forecasting captures
all dependencies indiscriminately and adds noise; channel clustering is too coarse for complex,
time-varying interactions. TimeFilter is a GNN-based framework that builds a graph from the input
sequence and filters out irrelevant spatial-temporal correlations while keeping the most critical ones
in a patch-specific manner, reaching state-of-the-art results on 13 real-world datasets.

## Implementation notes

- Node features: `Linear(patch_len, d_model)` on each channel's patches, plus an optional learnable `[enc_in, patches, d_model]` position term (`pos`).
- Filtering keeps, for each node, the `max(1, round(top_p * nodes))` strongest entries of its softmaxed affinity row (`top_p` in `(0, 1]`).
- Each `RegionExpert` passes messages over the filtered adjacency; the router mixes experts per node with softmax weights, and `last_moe_loss` measures expert load imbalance.
- `e_layers` filter layers, each recomputing the affinity from the current node features.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/HuZLLLC0XP25,
  author    = {Yifan Hu and Guibin Zhang and Peiyuan Liu and Disen Lan and Naiqi Li and Dawei Cheng and Tao Dai and Shu{-}Tao Xia and Shirui Pan},
  title     = {TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting},
  booktitle = {Forty-second International Conference on Machine Learning, {ICML} 2025},
  series    = {Proceedings of Machine Learning Research},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/hu25ac.html}
}
```
