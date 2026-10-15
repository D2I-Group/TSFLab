# AROpt — reference

## Paper

AROpt: An Optimization Method for Autoregressive Time Series Forecasting (arXiv 2602.02288, 2026-02; v3 2026-08). Section 3, Algorithm 1, Eqs. 3-5.

```bibtex
@article{li2026aropt,
  title   = {AROpt: An Optimization Method for Autoregressive Time Series Forecasting},
  author  = {Li, Zheng and Cheng, Jerry and Gu, Huanying Helen},
  journal = {arXiv preprint arXiv:2602.02288},
  year    = {2026}
}
```

## Implementation mapping

- `InvertedForecaster` is the iTransformer `f(.; theta)`: instance normalization, `DataEmbedding_inverted` (`embed` component), post-norm `transformer_encdec` encoder with full attention across variates, linear patch projector.
- `rollout` is Algorithm 1 (lines 3 and 7); `rollout_objective` is Eqs. (4)-(5) with `gamma = 0.5`, `beta = 0.1`: a later patch whose error falls below the previous one gets gradient weight `1 - 2 beta` (Eq. 3b).
- Official files read (nothing copied): `utils/tools.py` (`autoregressive_forcast`, `autoregressive_criterion`), `experiments/exp_long_term_forecasting.py`, `model/iTransformer.py`, `run.py`, `weather.sh` (`LizhengMathAi/AROpt`, revision `c18d9672`, MIT).

## Differences in detail

- Resolved from the official code: fed-back predictions are detached; the per-patch error is the mean error over the patch; defaults `training_k = 4` rollout steps, `gamma = 0.5`, `beta = 0.1`, and the iTransformer configuration (`d_model = 512`, 8 heads, 2 encoder layers, `d_ff = 2048`, dropout 0.1, GELU, instance normalization on) come from `run.py`.
- Horizon: the catalog window carries targets only up to `pred_len`, so the rollout covers the evaluation horizon with patches of `pred_len / rollout_steps` steps and evaluation uses the same number of calls. This is the paper's "AR = n" setting (for example a 96-step projector rolled out 4 times for a 384-step horizon); the "AR = 1" rows, where training rolls out beyond the evaluated horizon, are not reproduced.
- Rollout window: each call reads the last `seq_len` values of history plus predictions; the official code feeds only the previous predicted patch, which is the same when the patch is at least `seq_len` long.
- The per-patch error uses the configured observation criterion (MSE by default) rather than a fixed MSE.
- Calendar marks are not embedded as extra variate tokens (the official iTransformer appends time-feature tokens).
- Checked: patch length and divisibility, the rollout window update and detached feedback, the discounted objective against a hand computation and its gradient weights, and the reduction to the plain criterion for one rollout step. Reported benchmark numbers are not reproduction claims of this implementation.
