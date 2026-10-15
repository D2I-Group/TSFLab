# MultiPatchFormer — reference

## Paper

- **Title**: A multiscale model for multivariate time series forecasting
- **Venue**: Scientific Reports 2025 (no arXiv version)
- **Abstract (shortened)**: Most Transformer forecasters represent a series at a single scale, missing different time granularities, or ignore inter-series correlations. MultiPatchFormer divides the input into patches of several resolutions for multi-scale temporal modelling, follows the temporal encoder with a channel-wise encoder for inter-series interactions, and uses a multi-step linear decoder to reduce over-fitting and noise. It reports state-of-the-art error on seven real-world datasets and stronger generalizability.

## Implementation notes

- Each scale's token sequence is linearly interpolated to the largest token count among the scales before the per-scale embeddings (width `d_model / 4`) are concatenated.
- Encoder layers are pre-norm (`norm_first=True`).
- The semi-autoregressive head splits `pred_len` into `min(8, pred_len)` near-equal groups; group `g` reads the channel token plus all earlier groups' outputs.

## Citation

```bibtex
@article{naghashi2025multiscale,
  author  = {Vahid Naghashi and Mounir Boukadoum and Abdoulaye Banire Diallo},
  title   = {A Multiscale Model for Multivariate Time Series Forecasting},
  journal = {Scientific Reports},
  volume  = {15},
  number  = {1},
  year    = {2025},
  doi     = {10.1038/s41598-024-82417-4},
  url     = {https://doi.org/10.1038/s41598-024-82417-4}
}
```
