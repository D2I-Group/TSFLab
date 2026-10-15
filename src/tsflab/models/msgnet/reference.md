# MSGNet — reference

## Paper

- **Title**: MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting
- **Venue**: AAAI 2024 (arXiv 2401.00423, 2023-12)
- **Abstract (shortened)**: Inter-series correlations in multivariate series vary across time scales. MSGNet extracts salient periodic patterns by frequency-domain analysis to split the series into time scales, uses self-attention for intra-series dependence, and an adaptive MixHop graph convolution to learn inter-series correlations within each scale. The learned multi-scale correlations are explainable, and the paper reports strong generalization, including on out-of-distribution samples.

## Implementation notes

- The adjacency is `softmax(leaky_relu(E_s E_t, 0.01))` with low-rank factors of width `node_dim`; the small negative slope keeps both factors trainable before any edge scores positive.
- MixHop concatenates the zero-to-`gcn_depth`-hop states (`alpha * x + (1 - alpha) * A h`) and projects them with a bias-free linear layer, since a shared bias would be erased by the following node LayerNorm.
- Series whose length is not a multiple of a selected period are zero-padded at the end for segmentation and cropped after the branch.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/CaiLLFW24,
  author    = {Wanlin Cai and Yuxuan Liang and Xianggen Liu and Jianshuai Feng and Yuankai Wu},
  title     = {MSGNet: Learning Multi-Scale Inter-series Correlations for Multivariate Time Series Forecasting},
  booktitle = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI} 2024},
  pages     = {11141--11149},
  publisher = {{AAAI} Press},
  year      = {2024},
  doi       = {10.1609/AAAI.V38I10.28991}
}
```
