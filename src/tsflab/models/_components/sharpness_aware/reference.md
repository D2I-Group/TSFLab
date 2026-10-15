# sharpness_aware — reference

## Origin and granularity

Cut at the arithmetic only: the caller supplies `loss_fn`, which decides how a
batch becomes a loss, so the same component serves any model through the runner's
`ModelSpec.training_objective` hook. SAMformer is the only consumer today (its
`sharpness_aware_objective` method passes `rho`). The component was added with
SAMformer (commit `1a87e79b`). Choosing `rho`, reading the batch, and which
criterion is minimized stay in the model.

## Invariants and equivalence evidence

- A component check (added with the component) verified that `rho = 0` equals the plain loss,
  that the loss and parameter gradients equal a manual two-step SAM (ascent,
  gradient at `w + e`, restore) for both `adaptive=False` and `adaptive=True`, that
  parameters are unchanged and `aux` carries no grad, that the non-adaptive
  perturbation has norm `rho`, that the perturbed loss is not smaller than the
  clean loss for a small `rho`, and the two `ValueError` cases.
- no fixture: the comparison reference is the manual two-step procedure written
  in that check, not the official SAM/ASAM optimizer; any difference from that
  optimizer is expected to be floating-point order only (not measured).
- `samformer` model tests exercised the integration through its registered
  objective (the objective is SAM over the configured criterion; the spec declares
  the objective and components). These checks passed in the full suite run of
  2026-10-03 before the test suite was consolidated.

## Variants and options

`adaptive` only. Not provided: second-order terms, per-group `rho`, or
gradient-norm clipping.

## Related components

None overlaps in responsibility: no other component wraps a training loss. The
other custom-objective models (`dipelinear`'s SFALoss, `distdf`, `mmpd`) keep their
objectives model-local. The remaining `samformer` components, `revin` and
`channel_wise_linear`, are architecture blocks and unrelated to this wrapper.
