# bspline_basis — reference

## Origin and granularity

The recursion is the Cox-de Boor formula (Cox 1972; de Boor 1972). KAN (Liu et
al., arXiv 2404.19756) made B-splines on a uniform, order-extended grid the
learnable edge functions of a network. Two model-local copies of the same
recursion were consolidated:

- `arkan` (AR-KAN, arXiv 2509.02967): `BSplineKANLayer.basis`, one knot vector
  `[grid + 2 order + 1]` shared by all inputs (buffer `knots`).
- `tfkan` (TFKAN, arXiv 2506.12696): `BSplineKANLinear.b_splines`, one knot vector
  per input feature `[in, grid + 2 order + 1]` (buffer `grid`).

Only the basis evaluation is shared. The two layers are material variants and
stay local: AR-KAN has an output bias, a scalar spline scale per edge
(`spline_weight [out, in]`) times coefficients `coefficients [out, in, G + k]`
initialised `N(0, 0.1^2)`, and uniform base weights; TFKAN has no bias, per-edge
coefficients `spline_weight [out, in, G + k]` fitted by least squares to noise
times a Kaiming-initialised `spline_scaler [out, in]`, and a Kaiming base weight.
Merging them would rename state-dict keys or hide those differences behind flags.

## Invariants and equivalence evidence

- Both migrations are verbatim moves of the recursion: the same operations in the
  same order (indicator, then for each degree the left and right ratios and the
  blend), with the knot slices written `t[..., a:b]`, which for `[K]` and `[F, K]`
  knots is the same view as the former `t[a:b]` and `grid[:, a:b]`. Parameter and
  buffer names, registration order, and initialisation of both layers are
  untouched, so outputs, gradients and checkpoints are identical by construction.
- Equivalence against frozen copies of the former `BSplineKANLayer.basis` and
  `BSplineKANLinear.b_splines` was checked at extraction: exact equality
  (`torch.equal`) with the component and with the migrated layers, plus
  unchanged state-dict keys of both layers.
- Equation checks verified the partition of unity inside the interior knot span,
  local support, the closed-form linear hat for `order = 1` and the uniform cubic
  B-spline for `order = 3`, zero bases outside the knots, agreement of shared and
  per-feature knots, output shape and dtype, and gradient flow; `arkan` and
  `tfkan` had their own basis and layer checks. The frozen-copy and equation
  checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.

## Variants and options

Only the knot layout (`[K]` or `[F, K]`) and the degree vary. No grid update or
grid extension, no radial-basis (FastKAN) or Chebyshev bases, no closed right end.

## Related components

None in the catalog is a KAN layer; `mixer_block` and `channel_wise_linear` are
the linear mixing alternatives a KAN layer would replace.
