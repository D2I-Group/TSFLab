"""Loss registry and dynamic registration helpers."""

from __future__ import annotations

import importlib
from typing import Callable

import torch.nn as nn


class LossRegistry:
    """Registry mapping loss names to factory callables."""

    def __init__(self) -> None:
        self._losses: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str, factory: Callable[..., nn.Module]) -> None:
        """Register a loss factory by name."""
        self._losses[name.lower()] = factory

    def get(self, name: str) -> Callable[..., nn.Module]:
        """Get a loss factory by name.

        Raises
        ------
        KeyError
            If the loss is not registered.
        """
        key = name.lower()
        if key not in self._losses:
            raise KeyError(f"Loss '{name}' is not registered")
        return self._losses[key]

    def names(self) -> list[str]:
        """Return registered loss names."""
        return sorted(self._losses.keys())


LOSS_REGISTRY = LossRegistry()

LOSS_NAME_MAP = {
    "mse": "tsflab.experiments.losses",
    "mae": "tsflab.experiments.losses",
    "huber": "tsflab.experiments.losses",
    "freq_mae": "tsflab.experiments.losses.external",
    "freq_weighted_mae": "tsflab.experiments.losses.external",
    "masked_mae": "tsflab.experiments.losses.masked",
    "masked_mse": "tsflab.experiments.losses.masked",
    "masked_rmse": "tsflab.experiments.losses.masked",
    "quantile": "tsflab.experiments.losses.probabilistic",
    "nll_gaussian": "tsflab.experiments.losses.probabilistic",
}

_REGISTERED_LOSSES: set[str] = set()


def register_loss_by_name(name: str) -> None:
    """Import and register a loss using the name map.

    Parameters
    ----------
    name : str
        Loss name from the config.

    Raises
    ------
    KeyError
        If the loss name is not mapped.
    ModuleNotFoundError
        If the mapped module cannot be imported.
    AttributeError
        If the module has no register() function.
    """
    key = name.lower()
    if key in _REGISTERED_LOSSES:
        return
    module_name = LOSS_NAME_MAP.get(key)
    if module_name is None:
        available = ", ".join(sorted(LOSS_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Loss '{name}' is not mapped. Update LOSS_NAME_MAP in "
            f"tsflab.catalog.registry.losses. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Loss registry module not found: {module_name}. "
                "Expected module path in LOSS_NAME_MAP"
            ) from exc
        raise ImportError(
            f"Failed to import '{module_name}' due to missing dependency: {exc}"
        ) from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(
            f"Loss registry '{module_name}' must define a register() function"
        )
    register_fn()
    _REGISTERED_LOSSES.add(key)


def get_loss(name: str, **kwargs) -> nn.Module:
    """Resolve and construct a loss module by name."""
    register_loss_by_name(name)
    factory = LOSS_REGISTRY.get(name)
    return factory(**kwargs)
