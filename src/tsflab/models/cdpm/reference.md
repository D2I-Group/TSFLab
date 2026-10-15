# CDPM — reference

## Paper

- **Title**: Conditional Denoising Meets Polynomial Modeling: A Flexible Decoupled Framework for Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2024 (arXiv: 2024-10, revised to v7 2025-06)
- **arXiv**: https://arxiv.org/abs/2410.13253

## Implementation details

- `Model.statistics` fixes mean and standard deviation from the history and applies them to both
  windows (Sec. 4.2); the reused `series_decomposition` (kernel `moving_avg`, default 5) splits each
  into seasonal and trend parts (Eqs. 1-2).
- `PolynomialTrend` is the PTM (Eqs. 12-14): `Linear_origin(trend)` and
  `Linear_root(sign(trend) sqrt(|trend|))` mixed by a learnable `Linear(2, 1)` (`lambda_1`, `lambda_2`, bias).
- `ConditionalDenoiser` is the CDSM (Eqs. 4-11): a Conv1d embedding of the noisy seasonal future plus a
  learnable position table, `n_layers` blocks of step-conditioned AdaLN, a temporal Linear-ReLU mixer and
  a feature MLP with residuals, then a channel head. `prior` maps per-patch means and standard deviations
  of the historical seasonal part to the horizon patches with two MLPs and draws `mu + sigma * z`; the
  output is `w * denoised + (1 - w) * prior`.
- `teacher_forced_forecast` (the spec's `training_objective`) noises the true seasonal future at a random
  step (Eq. 3), predicts it directly, adds the trend forecast, de-normalises, and applies the configured
  loss (Eqs. 15-16).
- `forward` samples the seasonal part by deterministic DDIM (`eta = 0`) over all `K` steps with the clean
  estimate clipped to `[-1, 1]`; inference costs `diffusion_steps` denoiser passes per batch.
- Independent rewrite from the paper and the pinned official code (`zjt-gpu/CDPM@1d943bb0`, no license
  file, `NOASSERTION`). Paper/code conflicts are recorded as `issues` in `card.toml`.

## Citation

```bibtex
@misc{zhang2024cdpm,
  title         = {Conditional Denoising Meets Polynomial Modeling: A Flexible Decoupled Framework for Time Series Forecasting},
  author        = {Zhang, Jintao and Cheng, Mingyue and Tao, Xiaoyu and Liu, Zhiding and Wang, Daoyu},
  year          = {2024},
  eprint        = {2410.13253},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
