# DeepAir — reference

## Paper

- **Title**: Deep Distributed Fusion Network for Air Quality Prediction
- **Venue**: KDD 2018
- **Published**: 2018
- **arXiv**: N/A

## Abstract

DeepAir predicts the next 48 hours of air quality per monitoring station from air-quality, meteorology and
weather-forecast data. A spatial transformation converts sparse neighbouring readings into a consistent input
that simulates pollutant sources, and a deep distributed fusion network fuses heterogeneous urban data.
Deployed for 300+ Chinese cities, it beats 10 baselines on three years of nine-city data, with 2.4%, 12.2% and
63.2% relative gains on short-term, long-term and sudden-change prediction over the previous online system.

## Implementation mapping

- Inputs are `x_enc [B, seq_len, N]`, historical/future node covariates, and `spatial_mx [N, regions, N]`
  (passed as the dataset `adj_mx`); a 2-D adjacency is accepted as one region; without one a default
  projection over `spatial_regions` regions is built. The normalized output is `[B, pred_len, N]` in `(0, 1)`.
- Regional target history supplies the unavailable secondary-pollutant branch in the generic contract; exact
  coordinates, pollutant panels, terrain, and min-max statistics remain dataset responsibilities.
- The official `src/models/deepair.py` (unlicensed secondary reference) was inspected at the pinned revision;
  no external source was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/YiZWLZ18,
  author    = {Xiuwen Yi and Junbo Zhang and Zhaoyuan Wang and Tianrui Li and Yu Zheng},
  title     = {Deep Distributed Fusion Network for Air Quality Prediction},
  booktitle = {Proceedings of the 24th {ACM} {SIGKDD} International Conference on Knowledge Discovery {\&} Data Mining, {KDD} 2018, London, UK, August 19-23, 2018},
  pages     = {965--973},
  publisher = {{ACM}},
  year      = {2018},
  url       = {https://doi.org/10.1145/3219819.3219822},
  doi       = {10.1145/3219819.3219822}
}
```
