# dlinear — reference

## Origin and granularity

The scheme is DLinear from the LTSF-Linear paper; the moving-average decomposition
it uses is the Autoformer series decomposition. The backbone was reduced when
`series_decomposition` and `channel_wise_linear` were extracted, so this
component is now only the composition of the two with two projections. It is
shared because the same backbone feeds `dlinear`, `quantile_dlinear`
(followed by a quantile head), `latenttsf` (applied on latent channels
`c_in = d_model`), `lift` (as its time-domain branch, `c_in = enc_in`), and
`mtlinear` (a single-channel `c_in = 1` backbone for its `DLinear` layer type).
Left local: input shape validation, normalization, any head after the backbone,
and the latent autoencoder of `latenttsf`.

## Invariants and equivalence evidence

- A contract check (shared and individual) verified the state-dict keys
  (`linear` or `linears.0` stems, none under `decomposition`), output shape
  `[B, pred_len, C]` and dtype, that the output equals the seasonal plus trend
  projections of the decomposition, gradients reaching input and parameters,
  `ValueError` for a wrong `seq_len` and for an even `kernel_size`, and
  `kernel_size > seq_len` working; seeded outputs of both modes were pinned as
  regression values.
- An equation check verified that the `dlinear` model output equals seasonal
  projection plus trend projection of the decomposition, and a model-level check
  exercised `quantile_dlinear` through this backbone. These checks passed in the
  full suite run of 2026-10-03 before the test suite was consolidated.
- The regression values were recorded from the extracted backbone, not from a
  pre-extraction model, so equivalence with the original DLinear code is not
  backed by a stored pre-extraction reference.

## Variants and options

- `individual` per-channel vs shared linear layers.
- `kernel_size` controls the trend smoothness (25 is the paper default).
- For last-value centering instead of decomposition use `last_value_center`
  with `channel_wise_linear` (NLinear); for instance normalization use `revin`.

## Related components

`series_decomposition`, `channel_wise_linear`, `last_value_center`, `revin`,
`quantile_head` (used with it in `quantile_dlinear`).
- `fft_extrapolation_conv`: frequency-domain history-to-horizon alternative.
