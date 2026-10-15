# Koopa — reference

## Paper

Liu, Li, Wang, and Long, NeurIPS 2023 (arXiv 2305.18803, https://arxiv.org/abs/2305.18803).

Real-world series are non-stationary. Koopa uses Koopman theory: a Fourier Filter disentangles time-invariant and
time-variant components, and Koopman Predictors advance each with linear operators on learned measurement
functions, stacked in blocks that learn hierarchical dynamics. Context-aware operators are computed in the temporal
neighbourhood and can use incoming ground truth to extend the horizon. The deep residual structure removes the
binding reconstruction loss of earlier Koopman forecasters; Koopa is competitive while saving 77.3% training time
and 76.0% memory.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LiuLWL23,
  author    = {Yong Liu and Chenyu Li and Jianmin Wang and Mingsheng Long},
  editor    = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and
               Moritz Hardt and Sergey Levine},
  title     = {Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors},
  booktitle = {Advances in Neural Information Processing Systems 36, NeurIPS 2023},
  year      = {2023},
  url       = {http://papers.nips.cc/paper\_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html}
}
```
