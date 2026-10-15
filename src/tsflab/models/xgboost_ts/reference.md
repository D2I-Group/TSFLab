# XGBoostTS — reference

## Paper

XGBoost: A Scalable Tree Boosting System (KDD 2016, arXiv 1603.02754).

Describes XGBoost, a scalable end-to-end tree boosting system with a sparsity-aware algorithm for sparse data, a weighted quantile sketch for approximate tree learning, and insights on cache access patterns, data compression, and sharding. It scales beyond billions of examples with far fewer resources than existing systems.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/ChenG16,
  author    = {Tianqi Chen and Carlos Guestrin},
  editor    = {Balaji Krishnapuram and Mohak Shah and Alexander J. Smola and
               Charu C. Aggarwal and Dou Shen and Rajeev Rastogi},
  title     = {XGBoost: {A} Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd {ACM} {SIGKDD} International Conference on
               Knowledge Discovery and Data Mining, San Francisco, CA, USA, August
               13-17, 2016},
  pages     = {785--794},
  publisher = {{ACM}},
  year      = {2016},
  url       = {https://doi.org/10.1145/2939672.2939785},
  doi       = {10.1145/2939672.2939785}
}
```
