# ARKAN — reference

## Paper

AR-KAN: Autoregressive-Weight-Enhanced Kolmogorov-Arnold Network for Time Series Forecasting (arXiv 2509.02967, 2025-09; v3 2026-04). Sections II-B and III-B, Eqs. 5-12.

```bibtex
@article{zeng2025arkan,
  title   = {AR-KAN: Autoregressive-Weight-Enhanced Kolmogorov-Arnold Network for Time Series Forecasting},
  author  = {Zeng, Chen and Xu, Tiehang and Wang, Qiao},
  journal = {arXiv preprint arXiv:2509.02967},
  year    = {2025}
}
```

## Implementation mapping

- The AR memory is the buffer `ar_weights` (`h_i(n) = a_i delta(n - i)`), applied in `one_step`; `ar_fitted` marks whether `training_setup` has fitted it.
- `training_objective` trains one step ahead (`h = 1`): every horizon step is predicted from observed lags (`teacher_forced`); `forward` feeds predictions back.
- Official files read (nothing copied): `AR_predict_dataset.ipynb`, `AR_predict_functions.ipynb` (`ChenZeng001/AR-KAN`, revision `f9b5edc2`, MIT).

## Differences in detail

- Resolved from the official code: AR coefficients come from a no-intercept least-squares regression of the next value on the `p = 20` previous values over the training windows; the weighted lags enter the KAN in chronological order; the KAN is `pykan`'s `KAN(width=[20, 50, 1], grid=3, k=3)` trained with Adam (lr 1e-3) on one-step MSE without grid updates; test predictions are one step ahead from observed lags.
- KAN layer: a local B-spline implementation of Eqs. (5)-(6) (basis from `bspline_basis`; residual SiLU branch, per-edge spline scale, output bias, fixed uniform grid over `[-1, 1]`). Not reproduced: `pykan`'s symbolic branch, node and subnode affine scales, and its noise-based spline initialisation (here coefficients `N(0, 0.1^2)`, spline scale `1/sqrt(fan_in)`, base weights `U(+-1/sqrt(fan_in))`).
- Basis: the paper's appendix lists FastKAN (radial basis functions) for its KAN baselines while the official AR-KAN code and the error analysis use B-splines; B-splines are used here.
- Protocol: the paper is univariate and one step ahead; here channels are handled independently with per-channel AR coefficients and one shared KAN, and `H > 1` is forecast recursively at evaluation, so `pred_len > 1` is not the paper's protocol (`pred_len = 1` is).
- Before `training_setup` has fitted the memory, `ar_weights` are ones (the plain-KAN ablation). The training criterion is the run's configured loss (MSE in the paper).
- Checked: the B-spline basis (partition of unity, support, agreement with an explicit cubic B-spline), the KAN layer formula, least-squares recovery of known AR coefficients, Yule-Walker against the Eq. (12) autocorrelation, recursive rollout against repeated one-step calls, and teacher forcing. Reported benchmark numbers are not reproduction claims of this implementation.
