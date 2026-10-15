# CCDM — reference

## Paper

- **Title**: Channel-aware Contrastive Conditional Diffusion for Multivariate Probabilistic Time Series Forecasting
- **Venue**: arXiv preprint (under review)
- **Published**: 2024 (arXiv: 2024-10, v1)
- **arXiv**: https://arxiv.org/abs/2410.02168

## Implementation details

- `Model.window_statistics` normalizes history and target by the history's per-channel mean and
  standard deviation (`eps = 1e-5` inside the square root); samples are mapped back with the same statistics.
- `ChannelAwareDenoiser` (Sec. 3.1, Fig. 3): two channel-independent `DenseEncoder` stacks of TiDE
  residual blocks (`DenseResidual`, shared across channels) embed `x` (length L) and `y_k` (length H);
  their codes form one token per channel; `ChannelDiTBlock`s (self-attention over channels, adaLN
  shift/scale/gate from the `StepEmbedding` of `k`) mix them; `OutputDecoder` (adaLN, residual MLP,
  linear head) predicts the noise.
- `Model.training_loss` is Eq. 6: the noise-regression loss of Eq. 1 at an antithetic random step plus
  `contrast_weight` times the InfoNCE of Eq. 4; `contrastive_loss` scores the positive and
  `2 * n_negatives` negatives by denoising error under one shared noise draw (Eq. 3).
- Negatives (Appendix A.3): `patch_shuffle` permutes length-`shuffle_patch_len` patches of the target
  (a trailing remainder stays in place; fewer than two patches leaves the target unchanged);
  `amplitude_scale` multiplies each channel by a factor from `U[1.5, 2] u U[0, 0.5]`.
- `forward` runs `num_samples` ancestral DDPM trajectories over all `diffusion_steps` and returns
  empirical quantiles `[B, H, D, Q]`.

## Differences in detail

- Independent rewrite from Sections 2-3 and Appendix A.3-A.7 (Algorithm 1, Tables 5-6) after reading
  the pinned official code (`LSY-Cython/CCDM` at `a9a4998d`, no license file, `NOASSERTION`): `model.py`,
  `network.py`, `diffusion.py`, `embed.py`, `attention.py`, `utils/metric.py`,
  `data_producer/data_factory.py`, `data_producer/data_loader.py`, `run_I48_O96.py`. Nothing was copied or imported.
- Resolved from the official code: CiDM = `LayerNorm(Dropout_0.1(Linear(ReLU(Linear(x)))) + Linear(x))`;
  step embedding = 256 sinusoid features (cosine half first, max period 10000) then `Linear-SiLU-Linear`;
  attention has bias-free query/key/value projections, a biased output projection and dropout on the
  attention probabilities; DiT MLP is `Linear-GELU(tanh)-Linear` with ratio 1; LayerNorms before adaLN
  have no affine parameters (`eps = 1e-6`); noise parameterization with the DDPM posterior mean and
  variance; antithetic uniform step sampling; the per-sample denoising error is the mean over the window;
  the timestamp embedding is defined but unused, so marks are ignored.
- Defaults follow Table 5 / `run_I48_O96.py` for lookback 48, horizon 96 (hidden sizes 128, 2 DiT blocks,
  8 heads, quadratic schedule `1e-4 -> 0.5` over 50 steps) and Table 6 for ETTh1 (`lambda = 0.001`,
  64 negatives per augmentation, `tau = 0.1`).
- Paper/code conflicts are recorded as `issues` in `card.toml`: Eq. 3 scoring and the shared `eps'`
  follow the paper; the one-CiDM decoder follows the code.
- Point metrics use the runner's reduction of the quantile output (the paper reports the sample median).
- Training uses the CCDM objective through `training_objective`; the configured observation loss only
  drives validation and early stopping. The paper's optional two-stage schedule (pretrain without the
  contrastive term, then fine-tune) is a run-configuration choice and is not built in.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@misc{li2024ccdm,
  title         = {Channel-aware Contrastive Conditional Diffusion for Multivariate Probabilistic Time Series Forecasting},
  author        = {Li, Siyang and Chen, Yize and Xiong, Hui},
  year          = {2024},
  eprint        = {2410.02168},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
