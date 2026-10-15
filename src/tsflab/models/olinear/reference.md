# OLinear — reference

## Paper

- **Title**: OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain
- **Venue**: NeurIPS 2025 (arXiv 2505.08550, 2025-05)
- **Abstract (shortened)**: Temporal-forecast models encode and decode in the time domain, where entangled step-wise dependencies hurt; fixed transforms such as Fourier are dataset-independent. OLinear uses OrthoTrans, a data-adaptive orthogonal transform that diagonalizes the series' temporal Pearson correlation matrix, and NormLin, a linear layer with a normalized weight matrix for multivariate dependencies. NormLin outperforms multi-head self-attention at nearly half the FLOPs; OLinear reports state-of-the-art results on 24 benchmarks and 140 tasks, and NormLin improves Transformer forecasters as a plug-in.

## Implementation mapping

- OrthoTrans: persistent buffers `input_basis` (`seq_len x seq_len`) and `output_basis` (`pred_len x pred_len`); the lookback is mapped with `input_basis^T` along time and the decoded forecast with `output_basis`. `set_orthogonal_bases()` checks shapes and `Q^T Q = I` (tolerance `1e-4`) before copying.
- Embedding: each transformed step becomes a `d_model` token per channel (`Linear(1, d_model)`).
- CSL: `channel_in`, NormLin over channels (`softplus(W)` divided by its row sums), `channel_out`, residual plus LayerNorm.
- ISL: a residual GELU MLP (`d_model -> 2 d_model -> d_model`) plus LayerNorm.
- Decoder: `Linear(seq_len * d_model, pred_len)` per channel, then the output basis and optional affine RevIN denormalization.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2505-08550,
  author  = {Wenzhen Yue and Yong Liu and Haoxuan Li and Hao Wang and Xianghua Ying and Ruohao Guo and Bowei Xing and Ji Shi},
  title   = {OLinear: {A} Linear Model for Time Series Forecasting in Orthogonally Transformed Domain},
  journal = {CoRR},
  volume  = {abs/2505.08550},
  year    = {2025},
  doi     = {10.48550/ARXIV.2505.08550},
  url     = {https://doi.org/10.48550/arXiv.2505.08550}
}
```
