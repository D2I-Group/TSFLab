# AirCade — reference

## Paper

- **Title**: Spatiotemporal Causal Decoupling Model for Air Quality Forecasting
- **Venue**: ICASSP 2025 (arXiv 2505.20119, 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.20119

## Abstract

A causal-graph analysis shows that existing air-quality models do not fully capture the causal relations between the air quality index (AQI) and meteorological features.
AirCade combines a spatiotemporal module with knowledge embeddings to model AQI dynamics, a causal decoupling module that disentangles synchronous causality from past AQI and meteorology and passes it to future steps, and a causal intervention mechanism that represents uncertainty in future meteorology.
It reports over 20% relative improvement over prior models on an open-source air quality dataset.

## Runtime contract

The runtime consumes `x_enc [B, seq_len, N]`, historical covariates `[B, seq_len, N, cov_dim]`, and future covariates `[B, pred_len, N, cov_dim]`; raw six-column marks are converted to two calendar covariates. It returns `[B, pred_len, N]`.

## Citation

```bibtex
@inproceedings{ma2025aircade,
  author    = {Jiaming Ma and Guanjun Wang and Sheng Huang and Kuo Yang and Binwu Wang and Pengkun Wang and Yang Wang},
  title     = {Spatiotemporal Causal Decoupling Model for Air Quality Forecasting},
  booktitle = {2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2025)},
  pages     = {1--5},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.11099015}
}
```
