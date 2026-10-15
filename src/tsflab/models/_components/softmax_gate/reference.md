# softmax_gate — reference

## Origin and granularity

Extracted with PatchTSMixer, whose `AxisMixer` applies it after the MLP of each
mixing axis (patch, feature, and, only in `mix_channel` mode, channel), with the
gated axis permuted last and `dim` set to that axis width. Only the gate is shared; the MLP, norm, and residual stay in the
model. Distinct from `gated_fusion` (sigmoid convex blend of two tensors) and
`gated_dilated_conv` (tanh*sigmoid WaveNet unit).

## Invariants and equivalence evidence

- Gate weights are in (0, 1) and sum to one over the last axis, so `|out| <= |x|` elementwise.
- Contract (state-dict keys, shape/dtype, `ValueError` cases), invariant (weights sum to one), gradient, and seeded regression checks were run at extraction, with seeded reference values pinned as regression values.
- Model-level use: `patchtsmixer` structure checks verified that each `AxisMixer` computes `x + gate(mlp(norm(x)))` on the permuted axis (gated attention after each mixer's MLP; in `mix_channel` mode the inter-channel mixer comes before the patch mixer) and that `gate.score.weight` is `[dim, dim]`. No check against the Hugging Face implementation is cited here.
- These checks passed in the full suite run of 2026-10-03 before the test suite was consolidated; the consumer is now checked at admission (`tsf model verify`).

## Variants and options

Only `dim`. Not covered: gates whose output width differs from the input, sigmoid gates, temperature, or a residual form.

## Related components

`gated_fusion` (sigmoid blend of two tensors), `gated_dilated_conv` (tanh*sigmoid unit), `mixer_block` (the Chen et al. TSMixer block, a different model from the IBM TSMixer/PatchTSMixer; it has no gate).
