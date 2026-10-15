# MoSSL — reference

## Paper

- **Title**: Multi-Modality Spatio-Temporal Forecasting via Self-Supervised Learning
- **Venue**: IJCAI 2024 (arXiv 2405.03255), pages 2018-2026

## Differences in detail

- The paper (Secs. 3-4) and the official code at `2ca992ea` (`models/MoSSL.py`, `models/layers.py`, `traintest_MoSSL.py`, `lib/utils.py`) were read; the repository has no license and nothing was copied.
- Tensor input: TSFLab's spatiotemporal mode supplies `[B, T, N]`; the `(node, modality)` tensor is carried as `N = num_modalities * L` modality-major channels. With `num_modalities = 1` the modality attention attends only to the single modality and `L_c` is zero (no other modality to contrast). The runner may inject `num_nodes` and `adj_mx`; they are ignored.
- Layer detail: `AxisAttention` uses ReLU 1x1 query/key/value/output maps; `layers` layers consume exactly `1 + (k - 1)(2^layers - 1)` steps.
- Augmentation detail: modality relevance is scored from the original representation; a budget of `(node, modality)` cells is drawn with probability proportional to `1 - phi`; fully relevant cells are never masked; whole series of a cell are zeroed.
- GSSL predicts mixture memberships, means, and scales from the augmented representation and uses the Eq. 9 likelihood (summed per-channel log densities, log-sum-exp over components).
- MSSL fuses both views, summarizes each modality, and draws negatives by a random non-zero cyclic modality shift.
- Mask sampling uses the torch RNG instead of NumPy.
- The augmented view and both SSL losses run only in training mode; `self_supervised = false` disables them.
- Input scaling (the official global z-score) and the optimizer (Adam, lr 0.01, eps 1e-3, MultiStep 50/100, clipping 5, patience 15, batch 16) belong to the run configuration.
- Defaults are the paper's (`d_z = 48`, `K = 4`, 4 layers, kernel 2) and the code's mask ratio 0.1.
- No reference comparison against the official code was run.

## Checked properties

The modality-major layout; the receptive-length constraint; spatial and
modality attention; the modality-spanning gated convolution; the augmentation
(relevance distribution, whole-series cell masking, never masking fully
relevant cells, MoST embedding); the GSSL and MSSL objectives; and that the
self-supervised branch only adds `aux_loss` in training without changing the
forecast.

## Citation

```bibtex
@inproceedings{ijcai2024p223,
  title     = {Multi-Modality Spatio-Temporal Forecasting via Self-Supervised Learning},
  author    = {Deng, Jiewen and Jiang, Renhe and Zhang, Jiaqi and Song, Xuan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  pages     = {2018--2026},
  year      = {2024},
  doi       = {10.24963/ijcai.2024/223}
}
```
