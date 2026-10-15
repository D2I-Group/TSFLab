# channel_wise_linear — reference

## Origin and granularity

The shared-vs-individual `nn.Linear` pattern is the LTSF-Linear family's
temporal projection. It was extracted from the repeated local linear-layer
blocks of the LTSF-Linear models (commit `4fac489c`, "make model cards
canonical"). The cut is exactly the projection layer: normalization (`revin`,
`last_value_center`), decomposition (`series_decomposition`), layout permutes,
and any residual or mixing stay in the model. Direct consumers: `linear`,
`nlinear`, `rlinear`, `mtlinear`, `tsmixer`, `mtsmixer`, `rpmixer`, `distdf`,
`cosa`, `cyclenet`, `samformer`, `core`, plus the `composed` slot adapters;
`dlinear` uses it through the `dlinear` component. The catalog derives the
current consumer list from the code.

## Invariants and equivalence evidence

- Contract, invariant, gradient, and seeded numerical-regression checks for both
  the shared and the individual mode, with seeded reference values pinned as
  regression values.
- A repository contract check verified that shared and individual outputs and
  input gradients equal the direct `nn.Linear` formulation.
- A frozen TSMixer reference that uses `ChannelWiseLinear` as its projection
  pinned the state-dict keys of `tsmixer` (`projection.linear.*`).
- No pre-refactor tensor reference for the LTSF-Linear models; their equations
  were checked by equation checks of the compact local implementations. All of
  these checks passed in the full suite run of 2026-10-03 before the test suite
  was consolidated.

## Variants and options

- `individual=False`: one map shared by all channels (parameters independent of
  `channels`).
- `individual=True`: `channels` independent maps.
- Hidden layers, nonlinearities, dropout, and flatten-style heads are not
  provided: see `flatten_forecast_head` for patch-flatten heads.

## Related components

`dlinear` (two projections over a decomposition), `series_decomposition`,
`last_value_center`, `revin`, `flatten_forecast_head` (flatten-then-linear
head over patches; use it when the input is patch tokens rather than a raw
`[B, C, L]` series), `mixer_block` (adds nonlinearity and channel mixing
around such projections), `weight_set_router` (interpolates between the shared and
individual modes by mixing a few weight sets per channel).
- `channel_alignment`: width adapter between channel counts, versus a per-channel temporal projection.
- `patchtst`: channel-independent backbone alternative.
- `fft_extrapolation_conv`: parameter-efficient frequency-domain history-to-horizon alternative.
