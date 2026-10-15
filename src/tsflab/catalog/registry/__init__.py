"""Flat runtime catalogs without eager optional-runtime imports.

Importing a focused catalog such as :mod:`tsflab.catalog.registry.models` must stay
usable in metadata-only installations.  In particular, model and Agent catalog
commands do not require PyTorch merely because the registry package also
exposes the loss catalog.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "MODEL_CATALOG",
    "register_from_config",
]

_EXPORTS = {
    "DATASET_REGISTRY": ("tsflab.catalog.registry.datasets", "DATASET_REGISTRY"),
    "LOSS_REGISTRY": ("tsflab.catalog.registry.losses", "LOSS_REGISTRY"),
    "METRIC_REGISTRY": ("tsflab.catalog.registry.metrics", "METRIC_REGISTRY"),
    "MODEL_CATALOG": ("tsflab.catalog.registry.models", "MODEL_CATALOG"),
    "register_from_config": ("tsflab.catalog.registry.loader", "register_from_config"),
}


def __getattr__(name: str) -> Any:
    """Load only the requested catalog and cache the resolved export."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
