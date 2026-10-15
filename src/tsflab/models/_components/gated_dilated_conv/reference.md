# gated_dilated_conv — reference

## Origin and granularity

Extracted in commit `dd8d8196` ("gated_dilated_conv and
adaptive_node_embedding_adjacency") from near-identical copies in `wavenet`,
`gwnet`, `dfdgcn` and `mtgnn`, with equivalence tests for all four. The cut is
only pad plus gated product. Model-local: the conv modules themselves (their
parameters and state-dict names stay on each calling layer), residual and skip
convolutions, graph mixing, post-gate normalization, and the adjacency (MTGNN's
adjacency is a different operator and stays local).

## Invariants and equivalence evidence

- Contract check (at extraction): `causal_pad` shapes for rank 3 and 4, that the left pad is zero and the rest is the
  input, kernel 1 adds no padding, rank 2 raises `ValueError`; `gated_dilated_conv`
  length preservation with `Conv1d` and `Conv2d` `(1, k)`, dtype, `|y| < 1`,
  gradients to input and both convs, causality (perturbing future steps leaves earlier
  outputs unchanged); seeded output values were pinned as a regression value.
- Equivalence of each refactored model (`wavenet`, `gwnet` and `dfdgcn` together
  with the adaptive adjacency, `mtgnn`) against a frozen verbatim pre-extraction copy
  (same state-dict names, outputs, gradients, atol 1e-6) was checked at extraction;
  those frozen-copy tests passed in the full suite run of 2026-10-03 before the test
  suite was consolidated. The frozen copies were code, not stored tensors.

## Variants and options

The caller picks `Conv1d` or `Conv2d`, kernel width, dilation (e.g. exponential
across layers) and channel widths. A kernel of 1 yields zero padding. No
non-causal or symmetric padding option, no residual or skip output.

## Related components

`diffusion_conv` (the graph-mixing step that follows the gated unit in Graph WaveNet;
it mixes over nodes, this one over time), `adaptive_node_embedding_adjacency` (extracted
in the same commit; supplies the adjacency, not temporal gating), `gated_fusion`
(a learned sigmoid blend of two tensors, not a tanh x sigmoid temporal conv).
`composed` also reaches this component through its slot registry.
- `softmax_gate`: softmax feature gate on one tensor, not a tanh x sigmoid temporal convolution.
