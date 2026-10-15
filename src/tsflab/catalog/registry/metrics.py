"""Metric registry and dynamic registration helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable


class MetricRegistry:
    """Registry mapping metric names to callables."""

    def __init__(self) -> None:
        self._metrics: dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a metric function by name."""
        self._metrics[name] = func

    def get(self, name: str) -> Callable:
        """Get a metric function by name.

        Raises
        ------
        KeyError
            If the metric is not registered.
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
        return self._metrics[name]

    def names(self) -> list[str]:
        """Return registered metric names."""
        return sorted(self._metrics.keys())


METRIC_REGISTRY = MetricRegistry()

METRIC_NAME_MAP = {
    "mae": "tsflab.experiments.evaluation.metrics",
    "mse": "tsflab.experiments.evaluation.metrics",
    "rmse": "tsflab.experiments.evaluation.metrics",
    "mape": "tsflab.experiments.evaluation.metrics",
    "mspe": "tsflab.experiments.evaluation.metrics",
    "corr": "tsflab.experiments.evaluation.metrics",
    "rse": "tsflab.experiments.evaluation.metrics",
    "wape": "tsflab.experiments.evaluation.metrics",
    "smape": "tsflab.experiments.evaluation.metrics",
    "mase": "tsflab.experiments.evaluation.metrics",
    # Probabilistic metrics (Phase 1). These are NOT bare (pred, true)
    # callables — they are computed together by collect_prob_metrics and the
    # evaluator merges them into the metric dict directly. They are mapped here
    # so register_from_config() can register them without KeyError;
    # benchmark.evaluation.metrics.register() installs resolvable placeholders
    # for them so METRIC_REGISTRY.get(<name>) never raises for a mapped name.
    "crps": "tsflab.experiments.evaluation.metrics",
    "wql": "tsflab.experiments.evaluation.metrics",
    "coverage_80": "tsflab.experiments.evaluation.metrics",
    "width_80": "tsflab.experiments.evaluation.metrics",
}

_REGISTERED_METRICS: set[str] = set()


def register_metric_by_name(name: str) -> None:
    """Import and register a metric using the name map.

    Parameters
    ----------
    name : str
        Metric name from the config.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the metric name is not mapped.
    ModuleNotFoundError
        If the mapped module cannot be imported.
    AttributeError
        If the module has no register() function.
    """
    if name in _REGISTERED_METRICS:
        return
    module_name = METRIC_NAME_MAP.get(name)
    if module_name is None:
        available = ", ".join(sorted(METRIC_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Metric '{name}' is not mapped. Update METRIC_NAME_MAP in "
            f"tsflab.catalog.registry.metrics. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Metric registry module not found: {module_name}. Expected module path in METRIC_NAME_MAP"
            ) from exc
        raise ImportError(f"Failed to import '{module_name}' due to missing dependency: {exc}") from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(f"Metric registry '{module_name}' must define a register() function")
    register_fn()
    _REGISTERED_METRICS.add(name)
