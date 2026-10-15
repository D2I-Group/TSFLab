# TSMixer — reference

## Paper

TSMixer: An All-MLP Architecture for Time Series Forecasting (TMLR 2023, arXiv 2303.06053).

Simple univariate linear models can beat recurrent and attention models on common benchmarks. TSMixer extends them by stacking MLPs that mix along both the time and feature dimensions. It matches specialized state-of-the-art models on academic benchmarks and outperforms alternatives on the large M5 retail benchmark, which the authors attribute to using cross-variate and auxiliary information efficiently.

## Citation

```bibtex
@misc{chen2023tsmixer,
  author        = {Si-An Chen and Chun-Liang Li and Nate Yoder and
                   Sercan O. Arik and Tomas Pfister},
  title         = {TSMixer: An All-MLP Architecture for Time Series Forecasting},
  year          = {2023},
  eprint        = {2303.06053},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2303.06053}
}
```
