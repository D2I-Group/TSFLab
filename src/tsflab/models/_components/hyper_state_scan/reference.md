# hyper_state_scan — reference

## Origin and granularity

Added with `timepro` (automated intake), which is the only consumer. TimePro scans over the
variate axis with a scalar state, reshapes the state to a (time-patch, variate)
grid (patch-position height, variate width, patch features as channels), mixes
it locally, then reads it out. The module docstring states that
`GridStateMixer` replaces offset-based deformable convolution (DCNv4, needs a
custom CUDA kernel) by a fixed local receptive field, losing the learned sampling
offsets. Origin of the scan: Mamba (see `mamba`); the cut isolates only the two
paper-neutral pieces. Model-local in `timepro`: projections, `dt_rank`, `A` and `D`
parameters, the `C` read-out, the grid reshape, normalization and bidirectionality.

## Invariants and equivalence evidence

- Model-level checks (pre-consolidation suite): the scan was compared against an
  explicit loop recurrence, the mixer was checked to preserve the grid shape, and a
  `timepro` forward/backward test ran both inside the model (with `d_state=1`).
- Contract checks (at extraction): shape and dtype, the recurrence step by step,
  zero input giving zero state, `ValueError` for mismatched `u`/`b` shapes and a
  wrong-length `a`, gradients to `u` and `a`, the mixer's state-dict keys and
  `groups == channels`, channel independence (depthwise), and invalid
  `(channels, kernel_size)` pairs. Seeded reference outputs of the scan and of the
  mixer were pinned as regression values. These checks passed in the full suite run
  of 2026-10-03 before the test suite was consolidated.
- Equivalence with the `mamba` scan at `d_state=1` is not tested.

## Variants and options

Only `kernel_size` for the mixer. Not provided: multi-state (`d_state > 1`) scan
(use `mamba`), parallel or fused scan, learned sampling offsets or deformable
sampling, non-depthwise mixing.

## Related components

`mamba` (multi-state kernel-free selective scan with `C` read-out and `D` skip; this
module is the `d_state=1`, raw-state variant, not a replacement), `diffusion_conv`
and `graph_masked_attention` (other ways to mix over a variate axis, but with
supports or masks rather than a local conv grid).
