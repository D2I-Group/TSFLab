# DiPELinear — reference

## Paper

- **Title**: Disentangled Parameter-Efficient Linear Model for Long-Term Time Series Forecasting
- **Venue**: DASFAA 2026 (accepted; the arXiv listing and the official repository state this)
- **First posted**: 2024-11 (arXiv)
- **arXiv**: https://arxiv.org/abs/2411.17257

## Implementation details

- Local rewrite from Sections 3.2-3.6 (Eqs. 1-11). The official code (`wintertee/DiPE-Linear`, MIT) was
  inspected at `b26e2501` (`timeprophet/models/DiPE.py`, `timeprophet/utils/callbacks.py`, `scripts/DiPE.sh`)
  to resolve omissions; no source was copied. Outputs and SFALoss values match the official module on seeded
  random weights (single expert, three experts, history shorter and longer than the horizon) to 1e-5.
- Resolved from the official code, not stated in the paper: a guard band of about 1% of `L + H - 1` zeros on
  each side before the rfft; initialization (SFA/STA weights 1, IFM weight 1 at the DC bin and 0 elsewhere,
  bias 0, routing matrix standard normal); dropout 0.1 after SFA; mean/std instance normalization with the
  unbiased standard deviation clamped at 1e-7 (kept local because `revin` uses the biased variance plus
  epsilon); frequency-loss weights normalized to unit mean and, when history and horizon spectra differ in
  length, truncated or zero padded to the horizon's bins; the time-domain loss is the plain mean over all elements.

## Differences in detail

- From the paper: SFALoss normalizes by the mean absolute weight, equal to the paper's L1-norm form for
  positive weights (the official code divides by the signed mean). `dropout` is a parameter (hard-coded 0.1
  officially). Only the routed formulation is provided; the official `individual_f/t/c` switches and `t_loss`
  option (MAE time loss) are omitted, and the SFALoss time term is always MSE.
- From the official code: the temperature schedule is driven from `ModelSpec.training_setup` (training steps
  per epoch) and a step-counter buffer advanced by the training objective, so it follows the official linear
  per-epoch anneal without runner callbacks and resumes with checkpoints; in eval mode the temperature is the
  final value (the official code keeps the last training temperature during validation).
- The default preset uses `loss_alpha=0.9` (an official script value) and `num_experts=1`; per-dataset values
  (for example `num_experts=4` for Electricity and Weather, `use_revin=False` there) and the 720-step lookback
  are not preset. With `MS` features the loss uses the trailing channels' weights.
- No training was run; reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@misc{zhao2024dipelinear,
  title         = {Disentangled Parameter-Efficient Linear Model for Long-Term Time Series Forecasting},
  author        = {Yuang Zhao and Tianyu Li and Jiadong Chen and Shenrong Ye and Fuxin Jiang and Xiaofeng Gao},
  year          = {2024},
  eprint        = {2411.17257},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2411.17257}
}
```
