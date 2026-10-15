# GAGNN — reference

## Paper

- **Title**: Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting
- **Venue**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol. 18, No. 3, Article 55
- **Published**: 2024 (arXiv: 2021-08)
- **arXiv**: https://arxiv.org/abs/2108.12238

## Abstract

Nationwide city air quality forecasting must capture latent dependencies between geographically distant but highly correlated cities. GAGNN is a hierarchical model with a city graph (spatial dependencies) and a city-group graph (latent dependencies): a differentiable grouping network discovers city groups, a group correlation encoding module learns correlations between groups, and message passing models city-group dependencies. Experiments use a Chinese city air quality dataset.

## Citation

```bibtex
@article{DBLP:journals/tkdd/ChenXWH24,
  author       = {Ling Chen and
                  Jiahui Xu and
                  Binqing Wu and
                  Jianlong Huang},
  title        = {Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting},
  journal      = {{ACM} Trans. Knowl. Discov. Data},
  volume       = {18},
  number       = {3},
  pages        = {55:1--55:20},
  year         = {2024},
  url          = {https://doi.org/10.1145/3631713},
  doi          = {10.1145/3631713},
  timestamp    = {Sun, 19 Jan 2025 14:58:36 +0100},
  biburl       = {https://dblp.org/rec/journals/tkdd/ChenXWH24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
