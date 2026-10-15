# TimeO1 — reference

## Paper

Time-o1: Time-Series Forecasting Needs Transformed Label Alignment (Wang et al., NeurIPS 2025;
arXiv:2505.17847).

Temporal MSE faces two problems: label autocorrelation, which biases the label-sequence likelihood,
and a number of tasks that grows with the horizon. Time-o1 is a transformation-augmented objective that
maps the label sequence to decorrelated components of discriminated significance and trains the model
to align the most significant ones, mitigating autocorrelation and reducing the task count. It reaches
state-of-the-art results and is compatible with various forecast models.

## Implementation notes

- Labels are standardized per window and channel (mean and biased std over the horizon, eps 1e-5) before the SVD.
- The basis is the full right-singular-vector matrix per channel; `transform` multiplies the horizon by it, so components come in descending singular-value order.
- `rank_ratio` selects the leading fraction of components for the L1 term.
- `training_setup` collects `batch_y[:, -pred_len:]` over the whole training loader before fitting; until then `projection_ready` is false.

## Citation

```bibtex
@misc{wang2025timeo,
  author        = {Hao Wang and Licheng Pan and Zhichao Chen and Xu Chen and Qingyang Dai and Lei Wang and Haoxuan Li and Zhouchen Lin},
  title         = {Time-o1: Time-Series Forecasting Needs Transformed Label Alignment},
  year          = {2025},
  eprint        = {2505.17847},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.17847}
}
```
