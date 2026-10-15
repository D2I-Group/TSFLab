# Linear — reference

## Paper

Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (arXiv 2205.13504).

The paper questions Transformer-based long-term forecasting: permutation-invariant self-attention loses
temporal order information even with positional encoding. It introduces LTSF-Linear, a set of one-layer
linear models, which outperform sophisticated Transformer forecasters on nine real datasets, often by a large
margin, and studies how design elements affect temporal relation extraction.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZengCZ023,
  author    = {Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  title     = {Are Transformers Effective for Time Series Forecasting?},
  booktitle = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI} 2023},
  pages     = {11121--11128},
  publisher = {{AAAI} Press},
  year      = {2023},
  url       = {https://doi.org/10.1609/aaai.v37i9.26317},
  doi       = {10.1609/AAAI.V37I9.26317}
}
```
