# mixer_block — reference

## Origin and granularity

Extracted in commit `ea49cee5` ("mixer_block from TSMixer; audit mixer family")
verbatim from the model-local block in `tsmixer`, with attribute names unchanged;
the `tsmixer` model card records a clean-room implementation from paper
Appendix B.3.1-B.3.2 of the basic historical-target variant (arXiv 2303.06053).
The module docstring says to reuse the block only when mixing order, normalization
shape, and residual placement match exactly; paper-specific factorizations (other
mixers, low-rank channel bottlenecks, post-norm, extra hidden layers) stay local.
The final temporal projection stays in `tsmixer` (via `channel_wise_linear`).

## Invariants and equivalence evidence

- Contract check (at extraction): the exact state-dict key set, the `[6, 3]` norm
  and `[6, 6]` time-projection shapes, shape and dtype preservation, gradients to the
  input and every parameter, that zeroing `time_projection` and `feature_out` makes
  the block the identity, and seeded values pinned as a regression value.
- Equivalence against frozen copies of the pre-extraction `MixerBlock` and TSMixer
  `Model` (identical `state_dict()` keys and shapes, forward output, and gradients
  from a fixed seed) was checked at extraction; that frozen-copy test passed in the full suite run of 2026-10-03 before the test suite was consolidated.
- There is no test of non-default dropout behaviour or of shape-mismatch errors.

## Variants and options

None beyond the constructor arguments; activation is fixed to GELU, normalization
to joint `(L, C)` LayerNorm, and pre-norm order is fixed.

## Related components

`channel_wise_linear` (the temporal projection that follows the stack in
`tsmixer`), `gated_fusion` (learned blending of two branches rather than residual
mixing within one stream). `mtsmixer` (`FactorizedMixerBlock`) and `rpmixer`
(`RPMixerBlock`) keep their own mixer blocks, model-local, because their
factorization differs. `composed` only imports this block as a registration marker
(`# noqa: F401`), not as a used layer.
- `softmax_gate`: the gated-attention filter PatchTSMixer adds after its MLP mixers; this block has no gate.
