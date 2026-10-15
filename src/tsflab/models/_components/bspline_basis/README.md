---
name: "bspline_basis"
description: "Stateless Cox-de Boor evaluation of degree-k B-spline bases on shared [K] or per-feature [F, K] knots, the basis of B-spline KAN layers. Use for building a KAN layer or spline feature map on fixed knots; not for radial-basis or polynomial KANs, learnable knots, or as a complete KAN layer."
---

# bspline_basis

## What it does

`bspline_basis(x, knots, order)` evaluates every B-spline basis function of degree
`k = order` on the knot vector `t_0 <= ... <= t_{K-1}` at every input feature:

`B_{i,0}(u) = 1[t_i <= u < t_{i+1}]`,
`B_{i,d}(u) = (u - t_i) / (t_{i+d} - t_i) B_{i,d-1}(u) + (t_{i+d+1} - u) / (t_{i+d+1} - t_{i+1}) B_{i+1,d-1}(u)`,

returning the `K - k - 1` values `B_{i,k}(x_f)`. A B-spline KAN layer is
`y_o = sum_f w_b[o, f] silu(x_f) + sum_f sum_i c[o, f, i] B_i(x_f)` on top of these
bases; that parametrization stays in the consuming model.

## When to use

Use to build any B-spline KAN layer or spline feature map on fixed knots. Do not
use for radial-basis or polynomial KAN variants, for learnable knot positions that
need a gradient through the knots in degenerate spans, or as a full KAN layer (the
layer parametrization is model-specific).

## Interface

`bspline_basis(x, knots, order) -> Tensor`

- `x`: floating `[..., F]`, any leading shape.
- `knots`: `[K]` (shared by all features) or `[F, K]` (per feature), non-decreasing
  along the last axis; cast to `x.dtype`. Repeated knots inside an `order + 1` span
  divide by zero (no guard), as in both original copies.
- `order`: integer `k >= 0`; `order = 0` returns the indicator bases.
- Returns `[..., F, K - k - 1]` in `x.dtype` on `x.device`. Inputs outside
  `[t_0, t_{K-1})` get all-zero bases (the right end is open).
- Stateless: no parameters, buffers, or state-dict keys; the knots are owned and
  checkpointed by the caller.
