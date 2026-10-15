# empirical_quantiles — reference

## Origin and granularity

The interpolation rule is definition 7 of Hyndman and Fan, "Sample Quantiles in
Statistical Packages" (The American Statistician, 1996), the default of R,
NumPy and PyTorch. The helper was duplicated in six models and extracted here:
StaTS ("StaTS: Spectral Trajectory Schedule Learning for Adaptive Time Series
Forecasting with Frequency Guided Denoiser", `stats`), DynG-Diff ("A
State-Aware Dynamic Guidance Diffusion Framework for Probabilistic Time Series
Forecasting", `dyngdiff`), CCDM ("Channel-aware Contrastive Conditional
Diffusion for Multivariate Probabilistic Time Series Forecasting", `ccdm`),
NsDiff ("Non-stationary Diffusion For Probabilistic Time Series Forecasting",
ICML 2025, `nsdiff`), PPM ("Parametric Prior Mapping Framework for
Non-stationary Probabilistic Time Series Forecasting", ICML 2026, `ppm`) and
VolDyVAE ("Beyond Static Uncertainty: Modeling Temporal Uncertainty Dynamics
for Probabilistic Time Series Forecasting", `voldyvae`). None of these papers
defines its own quantile rule; each reports quantiles or CRPS of the sample set.
The cut is the samples-to-quantiles map only: the sampler, the number of
draws, the de-normalization of the draws, and the stored level buffer stay
model-local.

## Invariants and equivalence evidence

- Checks at extraction verified agreement with
  `torch.quantile(..., interpolation="linear")` for `dim=0`, `dim=1` and a
  negative `dim`, in float32 and float64, monotonicity for ascending levels,
  the levels 0 and 1 returning the sample minimum and maximum, `K = 1`, and
  gradients. Equivalence against frozen copies of the six pre-extraction
  helpers (the `stats`/`dyngdiff` form, the `ccdm` form, the `nsdiff` form, the
  `ppm` `sample_quantiles`, and the `voldyvae` form that sorts along `dim=1`)
  was checked under a fixed seed, with levels in the dtype each consumer
  passes: outputs identical, sample gradients equal up to 1 ulp (summation
  order). That frozen-copy test passed in the full suite run of 2026-10-03
  before the test suite was consolidated.
- Equivalence argument: `ppm` is a verbatim move (same body; `ppm` always
  passed `dim=1` explicitly). For the dim-0 consumers the only textual changes
  are a no-op `movedim(0, 0)`, a no-op cast of levels already cast to
  `samples.dtype` by the caller, and an explicit cast of the integer floor
  index that type promotion performed implicitly. `voldyvae` sorted before
  moving the sample axis; sorting along the axis before or after `movedim`
  gives the same values. Stateless, so no consumer state-dict key changes.
- Model-level checks in the pre-consolidation suite exercised the function
  through each model module: `ppm` and `voldyvae` against `torch.quantile`, and
  `stats`, `dyngdiff`, `ccdm`, `nsdiff` (forward equals the quantiles of the
  drawn samples).

## Variants and options

`dim` selects the sample axis (default 0). Not covered and kept local:
`ppgf`'s `group_edges` takes floor-rank order statistics of a flattened
training series (`x_(floor((K-1) k / G))`, no interpolation) to define
classification bins, not a forecast. Other interpolation rules (nearest,
midpoint, Harrell-Davis), weighted samples, and NaN-aware quantiles are not
provided.

## Related components

- `quantile_head`: parametric, non-crossing quantile output from features; this
  component is distribution-free and reads quantiles off samples.
- `gaussian_parameter_head`: Gaussian location/scale output; drawing from it and
  applying this component gives a Monte Carlo quantile estimate.
- `revin`: often undone on each draw before the quantiles are read (as in
  `voldyvae`).
