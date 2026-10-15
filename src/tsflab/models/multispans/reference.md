# MultiSPANS — reference

## Paper

- **Title**: MultiSPANS: A Multi-range Spatial-Temporal Transformer Network for Traffic Forecast via Structural Entropy Optimization
- **Venue**: WSDM 2024 (arXiv 2311.02880, 2023-11)

## Differences in detail

- Encoding tree: combine on any sibling pair, merge when at least one sibling is a module (a leaf counts as a singleton module), first index on ties, `max_tree_height = 3` by default; entropy uses the symmetrised, loop-free, positive finite weights. The official code builds the hierarchy with Infomap (map equation, copyleft package), which optimises a different objective.
- Masks: the diagonal is always allowed; leaves shallower than a level are singleton groups. Head count must exceed tree levels plus the adjacency head.
- Attention: a separate key projection and additive `-1e10` masking replace Eq. (7)'s reused query projection and Hadamard mask.
- Position routing (`official`): node encoding `D_s` on the temporal query and time encoding `D_t` on the spatial query, as in Sec. 3.3.1 and the code; `matched` swaps them as in Sec. 3.3.2.
- Mix-hop aggregation uses the transpose of the row-normalised kernel, as in the official `MixhopConv`.
- Transformer layer: post-norm with the position-encoded query in the attention residual, ReLU FFN with residual and a second BatchNorm (official default); the official `pre_norm=True` path is broken and not offered.
- No day-of-week or hour-of-day encoding `D_b` (official default `load_external = false`).
- Dense adjacency input only.
- Decoder: transposed convolution to `ceil(H / T) * T` steps, tanh, then time and channel linear maps.

## Citation

```bibtex
@inproceedings{zou2024multispans,
  title     = {MultiSPANS: A Multi-range Spatial-Temporal Transformer Network for Traffic Forecast via Structural Entropy Optimization},
  author    = {Zou, Dongcheng and Wang, Senzhang and Li, Xuefeng and Peng, Hao and Wang, Yuandong and Liu, Chunyang and Sheng, Kehua and Zhang, Bo},
  booktitle = {Proceedings of the Seventeenth ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2024}
}
```
