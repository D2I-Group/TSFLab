# CMoS — reference

## Paper

- **Title**: CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.19090

## Abstract

CMoS is a super-lightweight forecaster that, instead of learning shape embeddings, directly models the
spatial correlations between time-series chunks. Correlation Mixing captures diverse correlations with
minimal parameters, and optional Periodicity Injection speeds convergence. With as little as 1% of
DLinear's parameters it outperforms state-of-the-art models on multiple datasets, and its learned
weights are interpretable.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/SiPLPX25,
  author    = {Haotian Si and Changhua Pei and Jianhui Li and Dan Pei and Gaogang Xie},
  title     = {CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations},
  booktitle = {Forty-second International Conference on Machine Learning, {ICML} 2025, Vancouver, BC, Canada, July 13-19, 2025},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR} / OpenReview.net},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/si25a.html}
}
```
