# fft_extrapolation_conv — reference

## Origin and granularity

Extracted from the DiPE-Linear implementation (the only consumer today; see the
`dipelinear` card for the paper equation numbers). The cut is exactly the pad,
rfft, complex weighting, shift, irfft and slice; input normalization, the
attention maps applied before it, and the loss stay in the model, and the
weight-set routing matrix comes from `weight_set_router`. The guard band is an
implementation detail of the official code, not of the paper.

## Invariants and equivalence evidence

- `dipelinear` model-level checks verified, on random parameters, equality with
  an explicit circular time-domain convolution of the padded history plus the
  inverse-transformed bias; the DC average initialization; and linear
  weight-set mixing (one-hot mixing equals running a single-set layer on each
  channel group) with the `ValueError` cases for missing or wrongly shaped
  `mixing`. The seeded output of the mixing check was pinned as a regression
  value.
- A reference comparison reproduced the official DiPE-Linear implementation's
  forecasts through this layer from stored official reference values (single
  expert, three experts, and history shorter and longer than the horizon).
  These checks passed in the full suite run of 2026-10-03 before the test suite
  was consolidated.
- Not checked directly: gradients of the layer alone (the `dipelinear` training
  checks covered them indirectly), float64 inputs, and the constructor
  `ValueError`.

## Variants and options

One weight set shared by all channels (`num_sets=1`), or `num_sets` sets mixed per
channel through an externally supplied matrix (see `weight_set_router`). One set
per channel is `num_sets=channels` with an identity mixing matrix.

## Related components

`weight_set_router` (supplies `mixing`; its `mix_weight_sets` computes the same
per-channel combination for real weights, which this layer repeats inline for
complex ones), `channel_wise_linear` and `flatten_forecast_head` (dense,
time-domain history-to-horizon maps; this layer is the frequency-domain
parameter-efficient alternative), `dlinear` (dense linear baseline backbone).
`wavelet` and `haar_dwt1d` are fixed (not learned) multi-band transforms and are
not alternatives to this map.
