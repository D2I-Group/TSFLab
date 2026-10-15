---
name: "gaussian_parameter_head"
description: "Two linear heads giving a Gaussian location and a strictly positive softplus scale plus eps. Use for probabilistic forecasts trained with a Gaussian NLL per output element (DeepAR-style); not for correlated outputs, non-Gaussian likelihoods, bounded targets, or scales tied across horizons."
---

# gaussian_parameter_head

## What it does

`GaussianParameterHead(in_features, out_features)` returns an independent
Gaussian per output element: `loc = Linear_loc(x)` and
`scale = g(Linear_scale(x)) + eps`, with `g = softplus` or the algebraically equal
`log(1 + exp(.))`. The scale is strictly positive. It outputs parameters only; the
negative log-likelihood and sampling live in the loss and the model.

## When to use

Use when a model's output is an independent Gaussian per element and training uses
a Gaussian NLL. Do not use for correlated outputs, non-Gaussian likelihoods,
bounded targets, or when scale should be tied across horizons.

## Interface

`GaussianParameterHead(in_features: int, out_features: int, *, eps: float = 1e-6, scale_transform: "softplus" | "log1pexp" = "softplus")`

- `in_features`, `out_features` (int >= 1): last-axis widths of the input and of both outputs.
- `eps` (float > 0): added to the scale as a floor; `eps <= 0` raises `ValueError`.
- `scale_transform`: other values raise `ValueError`. `"log1pexp"` is `log(1 + exp(x))` computed naively and overflows to `inf` for large raw scale (above about 88 in float32); `"softplus"` is numerically stable.
- `forward(x [..., in_features]) -> (loc [..., out_features], scale [..., out_features])`; leading axes are arbitrary.
- State-dict keys: `loc_layer.weight`, `loc_layer.bias`, `scale_layer.weight`, `scale_layer.bias`. Stateless; float tensors on one device.
