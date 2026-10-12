"""First-order sharpness-aware minimization (SAM) as a training-loss wrapper.

SAM (Foret et al., ICLR 2021) minimizes the loss at the worst-case weights in a
rho-ball: ``w_adv = w + rho * g / ||g||`` with ``g = grad L(w)``; the optimizer
then steps ``w`` with ``grad L(w_adv)``. Because the gradient at ``w_adv`` is
what the base optimizer applies to ``w``, the update is reproduced by an
ordinary loss whose parameters are evaluated at the detached perturbation.
:func:`sharpness_aware_loss` returns exactly that loss, so any optimizer and the
standard ``loss.backward(); optimizer.step()`` loop implement SAM.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.func import functional_call


def sharpness_aware_loss(
    model: nn.Module,
    loss_fn: Callable[[Callable[..., torch.Tensor]], tuple[torch.Tensor | None, torch.Tensor]],
    rho: float,
    adaptive: bool = False,
    eps: float = 1e-12,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return ``(clean_aux, loss at w + e(w))`` for one SAM training step.

    ``loss_fn(run)`` must compute ``(aux, scalar_loss)`` by calling ``run(...)``
    exactly where it would call ``model(...)``; ``run`` is ``model`` itself for
    the ascent pass and a functional call with perturbed parameters for the
    descent pass. ``aux`` (for example a forecast) is passed through from the
    unperturbed pass, detached, and may be ``None``.

    ``e(w) = rho * g / (||g||_2 + eps)`` over all trainable parameters jointly;
    with ``adaptive=True`` it is ``rho * |w|^2 g / (|| |w| g ||_2 + eps)`` (ASAM
    form used by the SAM reference optimizer). ``e`` is detached, so the
    returned loss differentiates to ``grad L(w + e)`` with respect to ``w``
    (first-order SAM, no second derivatives). ``rho == 0`` is a single ordinary
    pass. No parameter is modified in place. Buffers are shared, not frozen:
    ``functional_call`` reuses the module's buffers, so a model with BatchNorm
    running statistics updates them in both the ascent and the descent
    forward pass (two updates per training step); use ``eval``-mode norms or
    stateless norms if that matters.
    """
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if rho == 0:
        aux, loss = loss_fn(model)
        return aux, loss
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not named:
        raise ValueError("model has no trainable parameters")
    aux, loss = loss_fn(model)
    grads = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    perturbed = _perturbed_parameters(named, grads, rho, adaptive, eps)

    def run(*args, **kwargs):
        return functional_call(model, perturbed, args, kwargs)

    _, adv_loss = loss_fn(run)
    return (None if aux is None else aux.detach()), adv_loss


def _gradient_norm(
    named: dict[str, nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
    adaptive: bool,
) -> torch.Tensor:
    """Joint L2 norm of ``g`` (or ``|w| g`` when ``adaptive``) over parameters with a gradient."""
    weighted = [
        None if g is None else ((p.detach().abs() if adaptive else 1.0) * g)
        for p, g in zip(named.values(), grads)
    ]
    return torch.sqrt(sum((w.detach() ** 2).sum() for w in weighted if w is not None))


def _perturbed_parameters(
    named: dict[str, nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
    rho: float,
    adaptive: bool,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Parameters at ``w + e(w)`` for every trainable parameter with a gradient."""
    norm = _gradient_norm(named, grads, adaptive)
    scale = rho / (norm + eps)
    perturbed = {}
    for (name, p), g in zip(named.items(), grads):
        if g is None:
            continue
        e = (p.detach() ** 2 if adaptive else 1.0) * g.detach() * scale
        perturbed[name] = p + e
    return perturbed


__all__ = ["sharpness_aware_loss"]
