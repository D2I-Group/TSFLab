---
name: "sharpness_aware"
description: "First-order sharpness-aware minimization (SAM/ASAM) as a loss at adversarially perturbed weights, usable with any optimizer via ModelSpec.training_objective. Use for papers that train with SAM (two passes per batch); not for DataParallel, parameter-mutating forwards, or unguarded float16."
---

# sharpness_aware

## What it does

SAM updates `w` with `grad L(w + e(w))`, where
`e(w) = rho * g / (||g||_2 + eps)` and `g = grad L(w)` over all trainable
parameters jointly. `sharpness_aware_loss(model, loss_fn, rho)` runs one ascent
pass to obtain `g`, detaches `e`, then evaluates the loss again with parameters
`w + e` through `torch.func.functional_call`. Backpropagating that loss gives,
with respect to the stored `w`, the gradient the reference SAM optimizer applies
after its second step, so the usual `loss.backward(); optimizer.step()` loop (any
base optimizer, weight decay included) realizes SAM without a wrapper optimizer.
With `adaptive=True` the ASAM form is used: the norm is `|| |w| * g ||_2` and
`e = rho * w^2 * g / (|| |w| * g ||_2 + eps)` (element-wise).

## When to use

Use for a model whose paper trains with SAM (as `samformer` does) and whose
runner already calls a custom objective. Do not use with `torch.nn.DataParallel`
(the runner raises for custom objectives there), with modules that mutate
parameters in `forward`, or when two independent dropout masks per step are
unacceptable. Under mixed precision the ascent gradient comes from the unscaled
loss and can underflow in float16. It costs two forward and two backward passes
per batch.

## Interface

`sharpness_aware_loss(model, loss_fn, rho, adaptive=False, eps=1e-12)`. The only
public symbol.

- `model`: `nn.Module`; only parameters with `requires_grad` are perturbed
  (parameters whose ascent gradient is `None` are left unperturbed). The
  function does not set train/eval mode; the runner calls it in training mode.
- `loss_fn(run)`: called twice (once with `run = model`, once with `run` a
  functional call that uses the perturbed parameters). It must call `run(...)`
  where it would call `model(...)` (only the forward call is redirected, not
  other methods) and return `(aux, loss)` with `loss` a scalar tensor; `aux` may
  be `None` or a tensor (for example the forecast).
- `rho` (float >= 0): neighborhood radius. `rho == 0` performs one ordinary pass
  (and skips the trainable-parameter check).
- `adaptive` (bool): ASAM-style scaling by `|w|` as in Purpose. `eps` (float):
  added to the gradient norm.
- Returns `(aux, loss)`: `aux` from the unperturbed pass, detached (`None` stays
  `None`); `loss` from the perturbed pass with autograd attached to the original
  parameters.
- Raises `ValueError` for negative `rho` and, when `rho > 0`, for a model without
  trainable parameters. No parameters, buffers, or state-dict keys are added; no
  in-place modification of parameters occurs. Buffers are not perturbed, but they are shared with
  `functional_call`, so a module that updates buffers in `forward` (for example
  BatchNorm running statistics) updates them in both the ascent and the descent pass, and stochastic layers such as dropout
  draw independent masks in the two passes.
