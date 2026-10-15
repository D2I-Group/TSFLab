# gaussian_parameter_head — reference

## Origin and granularity

The earliest history entry for this file is commit `4fac489c` ("make model
cards canonical"); its consumers are `deepar` and `gaussian_mlp`. The `log1pexp`
option exists to "preserve the official DeepAR expression for reference
comparison" (code comment); `deepar` uses the default `softplus` with
`eps=1e-6`. That `deepar` was the specific origin is inferred from this comment and
its consumer list; history does not record a different first author of the head.
Kept at the parameter-projection boundary: the recurrent trunk, the likelihood,
autoregressive sampling, and scale normalization of covariates remain in the models.

## Invariants and equivalence evidence

- Contract check (at extraction, both transforms): the four state-dict keys, `[2, 5, 3]` output shapes and dtype,
  `scale >= eps`, the softplus formula, gradients to the input and all parameters, and
  seeded values for each transform pinned as regression values; an error check
  covered the `eps` and `scale_transform` `ValueError`s and that softplus stays
  finite for a raw scale of 1000.
- Equation check: both transforms against their explicit formulas plus `eps`, strict
  positivity, and gradient finiteness; a `gaussian_mlp` model check verified the
  `GaussianMLP` head equations through the model.
- These checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- The `log1pexp` overflow for large raw scales stated in the Interface was never checked.

## Variants and options

`scale_transform` in `{"softplus", "log1pexp"}` and `eps`. Not covered: full
covariance, Student-t or negative-binomial heads, a fixed or shared scale.

## Related components

`quantile_head` (distribution-free alternative probabilistic output: input-conditioned
non-crossing quantiles instead of Gaussian `loc`/`scale`), `flatten_forecast_head` (point
forecast head with a different output type). `composed` lists this head in its `head`
schema literal but rejects it at validation (point output only), so it is not a working consumer.
