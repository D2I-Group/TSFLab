# InterPDN — reference

## Paper

Kong and Hong, AAAI 2026 (arXiv 2511.23260, https://arxiv.org/abs/2511.23260).

Scalar-output forecasters cannot express predictive uncertainty. interPDN (interleaved dual-branch Probability
Distribution Network) builds a discrete probability distribution per step and takes its expectation on a
predefined support set. A dual-branch design with interleaved support sets, plus coarse temporal-scale branches
for long-term trend, mitigates prediction anomalies; each branch's output is a self-supervised consistency
target for the other. Experiments on multiple real-world datasets show superior performance.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/KongH26,
  author    = {Linghao Kong and Xiaopeng Hong},
  editor    = {Sven Koenig and Chad Jenkins and Matthew E. Taylor},
  title     = {Time Series Forecasting via Direct Per-Step Probability Distribution Modeling},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026, Singapore},
  pages     = {22653--22661},
  publisher = {{AAAI} Press},
  year      = {2026},
  url       = {https://doi.org/10.1609/aaai.v40i27.39426},
  doi       = {10.1609/AAAI.V40I27.39426}
}
```
