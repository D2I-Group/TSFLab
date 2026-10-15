# CARD — reference

## Paper

- **Title**: CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting
- **Venue**: ICLR 2024 (arXiv 2305.12095, 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.12095

## Abstract

Channel-independent Transformers train robustly but ignore correlations among channels.
CARD introduces a channel-aligned attention structure that captures temporal correlations and dynamic dependence among variables, a token blend module that generates tokens at several resolutions, and a robust loss that weights the horizon by prediction uncertainty to reduce overfitting.
It outperforms prior methods on several long- and short-term datasets.

## Citation

```bibtex
@inproceedings{wang2024card,
  author    = {Xue Wang and Tian Zhou and Qingsong Wen and Jinyang Gao and Bolin Ding and Rong Jin},
  title     = {{CARD}: Channel Aligned Robust Blend Transformer for Time Series Forecasting},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR 2024)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=MJksrOhurE}
}
```
