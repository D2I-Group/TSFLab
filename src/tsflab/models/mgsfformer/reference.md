# MGSFformer — reference

## Paper

Yu et al., Information Fusion 113, 102607 (2025).

Air monitoring stations collect data at multiple sampling intervals (granularities) with distinct temporal patterns,
and stations are strongly spatiotemporally correlated. MGSFformer combines (1) a residual de-redundant block that
removes overlapping information across granularities, (2) a spatiotemporal attention block over stations and time
steps, and (3) a dynamic fusion block that weights and merges the per-granularity predictions. It beats 11
baselines by about 5% on three real-world air quality datasets.

## Citation

```bibtex
@article{DBLP:journals/inffus/YuWWSSYX25,
  author  = {Chengqing Yu and Fei Wang and Yilun Wang and Zezhi Shao and Tao Sun and Di Yao and Yongjun Xu},
  title   = {MGSFformer: {A} Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction},
  journal = {Inf. Fusion},
  volume  = {113},
  pages   = {102607},
  year    = {2025},
  url     = {https://doi.org/10.1016/j.inffus.2024.102607},
  doi     = {10.1016/J.INFFUS.2024.102607}
}
```
