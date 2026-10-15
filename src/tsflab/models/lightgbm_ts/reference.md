# LightGBMTS — reference

## Paper

Ke et al., NeurIPS 2017 (conceptual background only).

GBDT implementations such as XGBoost and pGBRT scan all instances per feature to estimate split gains, which is
slow for high-dimensional, large data. LightGBM adds Gradient-based One-Side Sampling (GOSS), which keeps
large-gradient instances and samples the rest, and Exclusive Feature Bundling (EFB), which greedily bundles
mutually exclusive features. It trains up to over 20x faster than conventional GBDT at almost the same accuracy.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/KeMFWCMYL17,
  author    = {Guolin Ke and Qi Meng and Thomas Finley and Taifeng Wang and Wei Chen and
               Weidong Ma and Qiwei Ye and Tie{-}Yan Liu},
  title     = {LightGBM: {A} Highly Efficient Gradient Boosting Decision Tree},
  booktitle = {Advances in Neural Information Processing Systems 30 (NeurIPS 2017)},
  pages     = {3146--3154},
  year      = {2017},
  url       = {https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html}
}
```
