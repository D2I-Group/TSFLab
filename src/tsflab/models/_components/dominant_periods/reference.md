# dominant_periods — reference

## Origin and granularity

The routine is the period-discovery step of TimesNet (FFT_for_Period) and is
reused by `msgnet` for its scale-graph branches. The git history does not
record the first extraction commit of this component (the file was moved
during the colocation refactor `33ea2050`); the origin is taken from its two
consumers and from the name of its former contract check. It is cut at the pure function: spectrum,
top-k, period arithmetic. Model-local: how periods are used (TimesNet folds the
series into a 2D image of shape `[B, ceil(T/p), p, C]` and applies an Inception
block; MSGNet builds one graph branch per period), and the weighting of branch
outputs (softmax over the amplitudes in both consumers).

## Invariants and equivalence evidence

- A contract check verified shapes, integer dtype, the recovered periods
  `[6, 12]` of a two-sine input (strongest first), gradient to `x`, `k` out of
  range (0 and 13 for `T=24`) and non-3-D input errors; seeded reference values
  were pinned as a regression value.
- A repository contract check compared periods, amplitudes, and the gradient of
  the amplitudes against an inline reference implementation of the TimesNet /
  MSGNet routine.
- A `timesnet` equation check verified that a sine of period 4 over length 16
  yields period `[4]` and amplitudes of shape `[1, 1]`.
- These checks passed in the full suite run of 2026-10-03 before the test suite
  was consolidated. No pre-refactor model reference: no stored before/after
  outputs of `timesnet` or `msgnet`; the inline reference comparison above is
  the equivalence evidence.

## Variants and options

Only `k`. There is no option for per-sample periods, windowing, or
interpolation of fractional periods: the period is the integer quotient
`T // f`, so non-divisor frequencies are rounded down.

## Related components

- `harmonic_energy_gate`: periodicity as a spectral energy ratio (a soft weight,
  not integer periods).
- `frequency_band_sampler`: selects contiguous FFT bands by depth; does not turn
  frequencies into periods.
- `periodic_query_bank`: gathers a learnable table for a fixed known period instead
  of a discovered one.
- `spectral_descriptor`: per-window spectral summary (entropy, band-energy ratios)
  from the same kind of channel-averaged spectrum, with no period output.
- `periodic_alibi_bias`: its periods can parameterize the bias slopes.
