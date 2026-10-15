"""B-spline basis values by the Cox-de Boor recursion.

Paper-neutral building block of B-spline Kolmogorov-Arnold network (KAN) layers:
for a knot vector ``t_0 <= ... <= t_{K-1}`` and degree ``k`` it evaluates the
``K - k - 1`` basis functions ``B_{i,k}`` at every input,

    B_{i,0}(u) = 1[t_i <= u < t_{i+1}],
    B_{i,d}(u) = (u - t_i) / (t_{i+d} - t_i) B_{i,d-1}(u)
               + (t_{i+d+1} - u) / (t_{i+d+1} - t_{i+1}) B_{i+1,d-1}(u).

The knots may be shared by all features (``[K]``) or given per feature
(``[features, K]``); they are broadcast against ``x [..., features]``. The layer
parametrization (base branch, spline coefficients and scales, bias,
initialization) belongs to the consuming model.
"""

from __future__ import annotations

import torch


def bspline_basis(x: torch.Tensor, knots: torch.Tensor, order: int) -> torch.Tensor:
    """Degree-``order`` B-spline bases of ``x [..., F]`` -> ``[..., F, K - order - 1]``.

    ``knots`` is ``[K]`` (shared) or ``[F, K]`` (per feature), non-decreasing along
    its last axis with distinct consecutive values over every ``order + 1`` span; it
    is cast to ``x.dtype``. Inputs outside ``[t_0, t_{K-1})`` get all-zero bases.
    """
    t = knots.to(x.dtype)
    u = x.unsqueeze(-1)
    values = ((u >= t[..., :-1]) & (u < t[..., 1:])).to(x.dtype)
    for degree in range(1, order + 1):
        left = (u - t[..., : -(degree + 1)]) / (t[..., degree:-1] - t[..., : -(degree + 1)])
        right = (t[..., degree + 1 :] - u) / (t[..., degree + 1 :] - t[..., 1:-degree])
        values = left * values[..., :-1] + right * values[..., 1:]
    return values


__all__ = ["bspline_basis"]
