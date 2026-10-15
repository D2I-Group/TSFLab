# Crossformer — reference

## Paper

- **Title**: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
- **Venue**: ICLR 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract

Transformer forecasters mostly model cross-time dependency and omit cross-dimension (cross-variable)
dependency. Crossformer embeds the input into a 2D vector array with Dimension-Segment-Wise (DSW) embedding,
captures cross-time and cross-dimension dependency efficiently with a Two-Stage Attention (TSA) layer, and
builds a Hierarchical Encoder-Decoder (HED) that uses information at several scales for the forecast.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/ZhangY23,
  author    = {Yunhao Zhang and Junchi Yan},
  title     = {Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting},
  booktitle = {The Eleventh International Conference on Learning Representations, {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher = {OpenReview.net},
  year      = {2023},
  url       = {https://openreview.net/forum?id=vSVLM2j9eie}
}
```
