# Aurora — reference

## Paper

- **Title**: Aurora: Towards Universal Generative Multimodal Time Series Forecasting
- **Venue**: ICLR 2026 (arXiv 2509.22295, 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.22295

## Abstract

Unimodal time-series foundation models ignore domain knowledge held in text, and end-to-end multimodal models do not support zero-shot cross-domain inference.
Aurora is a multimodal time-series foundation model pretrained on a cross-domain multimodal corpus; it tokenizes, encodes and distils text or image knowledge and injects it through Modality-Guided Multi-head Self-Attention.
Multimodal representations generate conditions and prototypes of future tokens for Prototype-Guided Flow Matching (generative probabilistic forecasting); results are state of the art on TimeMMD, TSFM-Bench, ProbTS, TFB, and EPF.

## Citation

```bibtex
@article{wu2025aurora,
  author  = {Xingjian Wu and Jianxin Jin and Wanghui Qiu and Peng Chen and Yang Shu and Bin Yang and Chenjuan Guo},
  title   = {Aurora: Towards Universal Generative Multimodal Time Series Forecasting},
  journal = {CoRR},
  volume  = {abs/2509.22295},
  year    = {2025},
  doi     = {10.48550/ARXIV.2509.22295}
}
```
