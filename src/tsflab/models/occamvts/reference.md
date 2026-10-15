# OccamVTS — reference

## Paper

- **Title**: OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting
- **Venue**: AAAI 2026 (arXiv 2508.01727, 2025-08)
- **Abstract (shortened)**: Large vision models help forecasting through visual representations, but 99% of their parameters are unnecessary: time series align with low-level textural features, not high-level semantics, which can hurt accuracy. OccamVTS distills the essential 1% of predictive information from pretrained vision models (privileged teachers) into lightweight networks via pyramid-style feature alignment with correlation and feature distillation. It reports state-of-the-art accuracy with 1% of the parameters, especially in few-shot and zero-shot settings.

## Implementation mapping

- Temporal branch: RevIN, then per-channel overlapping patches projected linearly to `d_model` with a learned positional parameter, then a pre-norm `nn.TransformerEncoder` (feed-forward width `2 * d_model`).
- Visual branch: four channels per series (raw; rFFT magnitude interpolated to `seq_len` and max-normalized; sine and cosine of `2 pi t / period`), two 1-D convolutions (kernels 5 and 3, GELU), then adaptive average pooling to the patch count.
- Fusion: multi-head cross-attention with temporal tokens as queries and visual features as keys/values, residual plus LayerNorm, mean over tokens, linear head to `pred_len`, RevIN denormalization.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LyuZRLWXL26,
  author    = {Sisuo Lyu and Siru Zhong and Weilin Ruan and Qingxiang Liu and Qingsong Wen and Hui Xiong and Yuxuan Liang},
  title     = {OccamVTS: Distilling Vision Models to 1{\%} Parameters for Time Series Forecasting},
  booktitle = {Fortieth {AAAI} Conference on Artificial Intelligence, {AAAI} 2026},
  pages     = {24216--24225},
  publisher = {{AAAI} Press},
  year      = {2026},
  doi       = {10.1609/AAAI.V40I29.39601},
  url       = {https://doi.org/10.1609/aaai.v40i29.39601}
}
```
