"""Registry bootstrap helpers."""

from __future__ import annotations

from tsflab.catalog.registry.datasets import register_dataset_by_name
from tsflab.catalog.registry.metrics import register_metric_by_name
from tsflab.catalog.registry.models import MODEL_CATALOG


def register_from_config(config) -> None:
    """Register datasets, models, and metrics referenced by a config.

    Parameters
    ----------
    config : RootConfig
        Validated config object.

    Returns
    -------
    None
    """
    register_dataset_by_name(config.dataset.name)
    MODEL_CATALOG.get(config.model.name)
    if config.evaluation.metrics:
        for metric_name in config.evaluation.metrics:
            register_metric_by_name(metric_name)
