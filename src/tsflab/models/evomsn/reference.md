# EvoMSN — reference

## Implementation mapping

- `global_periods` (Eq. 1): spectra averaged per batch over the training
  loader; DC and bins below `seq_len // (2 * pred_len) + 1` (periods longer than
  about `2 * pred_len`) zeroed.
- `normalize` (Eqs. 2, 6): for each period the window is padded with its own
  trailing segment before slicing.
- `ScaleStatisticsPredictor` (Eq. 5): two-layer perceptrons, 512 hidden units
  per input (slice statistics and raw window); the mean branch has Tanh hidden,
  identity output, is centered on the window mean and recombined with learnable
  per-channel weights initialised to one; the std branch uses ReLU hidden and
  output. `denormalize` applies Eq. (8).
- `ensemble_weights` (Eqs. 9-10): local amplitude at bin `seq_len // p`,
  normalized over the `k` periods.
- `pretrain`: statistics predictors fitted with MSE and frozen; in the backbone
  stage only the backbone is optimized.

## Differences in detail

- Sources read: `models/multiscale_stat_prediction.py`, `exp/exp_MSN.py`,
  `models/UnifiedModel.py`, `run_longExp.py`.
- From the official code (not in the paper): the period-search details, MLP
  widths, centered mean branch with per-channel weights, normalization constant
  `1e-5`, and from `run_longExp.py` `top_k = 4`, 5 statistics-pretraining
  epochs, statistics learning rate `1e-4`.
- The online protocol of the paper's main tables is not reproduced.
- `pretrain` replaces the placeholder periods with the data-selected ones and
  rebuilds the statistics predictors.
- Checked: padding and slicing, Eqs. (3)-(4) statistics, the admissible
  frequency mask and period selection on a synthetic spectrum, slice
  standardization, Eq. (8), Eqs. (9)-(10), the forward ensemble, the statistics
  loss, and that pretraining freezes the statistics predictors. Reported
  benchmark numbers are not reproduction claims of this implementation.

## Citation

Qin, D., Li, Y., Chen, W., Zhu, Z., Wen, Q., Sun, L., Pinson, P., Wang, Y. "Evolving Multi-Scale Normalization for Time Series Forecasting under Distribution Shifts." arXiv:2409.19718 (2024).
