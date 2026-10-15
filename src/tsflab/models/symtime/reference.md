# SymTime — reference

## Paper

Synthetic Series-Symbol Data Generation for Time Series Foundation Models (Wang, Wu, Li, Wang, Zhang;
NeurIPS 2025; arXiv:2510.08445).

Time series foundation models suffer from training-data scarcity and imbalance. Inspired by complex
dynamic system theory, the paper generates unlimited high-quality series paired with symbolic
expressions and pre-trains SymTime, which enhances series representations with symbolic information.
Fine-tuned, it is competitive on five major time series analysis tasks, rivaling foundation models
pre-trained on real-world data.

## Implementation notes

- Patch tokens get a learned positional parameter before the encoder; `d_model` must be divisible by `num_heads`.
- `trend_kernel` must be a positive odd integer (moving-average window of the decomposition).
- `periodic_head` maps `num_patches * d_model` to `pred_len`; `trend_head` maps `seq_len` to `pred_len`.
- RevIN wraps the whole model.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2510-08445,
  author  = {Wenxuan Wang and Kai Wu and Yujian Betterest Li and Dan Wang and Xiaoyu Zhang},
  title   = {Synthetic Series-Symbol Data Generation for Time Series Foundation Models},
  journal = {CoRR},
  volume  = {abs/2510.08445},
  year    = {2025},
  doi     = {10.48550/ARXIV.2510.08445}
}
```
