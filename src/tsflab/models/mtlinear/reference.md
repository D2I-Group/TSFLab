# MTLinear — reference

## Paper

- **Title**: A Multi-Task Learning Approach to Linear Multivariate Forecasting
- **Venue**: AISTATS 2025 (arXiv 2502.03571, 2025-02)

## Differences in detail

- Implementation: independent local rewrite from Sections 4.2 to 4.4 (Eqs. 7 and 8). The official repository (`azencot-group/MTLinear`, revision `7d0d9fb71d0e3652829ed28dda5acc558b27f528`) has no LICENSE file, so it was read only to resolve omissions (`models/MTLinear.py`, `utils/grad_manip.py`, `exp/exp_main.py`, `layers/Linear_layers.py`); nothing was copied.
- Resolved from the official code: complete-linkage agglomeration on `1 - corr`, merging only below the threshold; the penalty uses the batch mean absolute error per group (`H_i` over horizon and batch, `K_j` over variates and batch); zero errors are replaced by 1 before inversion; the weights are detached constants; the horizon-only penalty is scaled by `group size / enc_in`, and the variate-only penalty is broadcast over the horizon; DLinear, NLinear, Linear, and RLinear heads are shared across the variates of a group.
- With both penalties off, the configured criterion is used instead of the penalized squared error.
- Differences from the official code:
  1. `cluster_dist` is always a distance threshold; the official code treats an integer value as a fixed cluster count and has no `max_clusters` cap (here the grouping is re-cut to at most `max_clusters` groups).
  2. The grouping is fitted once through `ModelSpec.training_setup` from the full scaled training series of the loader's dataset (falling back to the newest step of every training window, which drops the first lookback, for loaders without a series-backed dataset), not at model construction. Before it runs, variates are assigned round-robin; NaN correlations (constant series) are set to 0.
  3. The official multi-block early stopping (`multi_early_stopping`, one best checkpoint per group) is not implemented; the runner's single early-stopping rule applies.
  4. With `MS` features the weights use the trailing channels' groups.
  5. `kernel_size` is a parameter (official 25, must be odd); per-dataset settings (for example sequence length 36 and learning rate 0.01 for the illness grid) are not preset.
- Clustering was checked against a complete-linkage reference; no training was run. Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{nochumsohn2025mtlinear,
  title         = {A Multi-Task Learning Approach to Linear Multivariate Forecasting},
  author        = {Nochumsohn, Liran and Zisling, Hedi and Azencot, Omri},
  booktitle     = {Proceedings of the 28th International Conference on Artificial Intelligence and Statistics},
  year          = {2025},
  eprint        = {2502.03571},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2502.03571}
}
```
