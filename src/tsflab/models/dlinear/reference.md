# DLinear — reference

## Paper

- **Title**: Are Transformers Effective for Time Series Forecasting?
- **Venue**: AAAI 2023 (Oral)
- **Published**: 2023 (arXiv: 2022-05)
- **arXiv**: https://arxiv.org/abs/2205.13504

## Abstract

The paper questions Transformer-based long-term forecasting (LTSF): permutation-invariant self-attention
inevitably loses temporal information even with positional encodings and sub-series tokens. A set of
embarrassingly simple one-layer linear models, LTSF-Linear, outperforms sophisticated Transformer LTSF models on
nine real-life datasets in all cases, often by a large margin, and empirical studies probe how design elements
affect temporal relation extraction.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZengCZ023,
  author    = {Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  title     = {Are Transformers Effective for Time Series Forecasting?},
  booktitle = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI} 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February 7-14, 2023},
  pages     = {11121--11128},
  publisher = {{AAAI} Press},
  year      = {2023},
  url       = {https://doi.org/10.1609/aaai.v37i9.26317},
  doi       = {10.1609/AAAI.V37I9.26317}
}
```
