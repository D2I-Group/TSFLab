# AMRC — reference

## Paper

- **Title**: Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency
- **Venue**: NeurIPS 2025 (arXiv 2510.19980, 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.19980

## Abstract

Systematic experiments show that appropriately truncating historical data can improve accuracy, so models learn substantial redundant features (noise, irrelevant fluctuations) from long inputs.
Based on information bottleneck theory, AMRC adds a dynamic masking loss that adaptively identifies discriminative temporal segments to guide training, and a representation consistency constraint that stabilizes mappings among inputs, labels, and predictions.
Experiments show it suppresses redundant feature learning and improves performance.

## Citation

```bibtex
@article{liang2025amrc,
  author  = {Renzhao Liang and Sizhe Xu and Chenggang Xie and Jingru Chen and Feiyang Ren and Shu Yang and Takahiro Yabe},
  title   = {Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency},
  journal = {CoRR},
  volume  = {abs/2510.19980},
  year    = {2025},
  doi     = {10.48550/ARXIV.2510.19980}
}
```
