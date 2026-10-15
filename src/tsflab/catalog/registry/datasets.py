"""Dataset registry and dynamic registration helpers."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

TaskMode = Literal["time_series", "spatiotemporal", "covariate"]
StorageMode = Literal["file", "directory", "selector"]
TASK_MODE_ORDER: tuple[TaskMode, ...] = (
    "time_series",
    "spatiotemporal",
    "covariate",
)


def ordered_task_modes(task_modes: frozenset[TaskMode]) -> tuple[TaskMode, ...]:
    """Return task modes in the one stable public presentation order."""
    unknown = task_modes.difference(TASK_MODE_ORDER)
    if unknown:
        raise ValueError(f"unknown dataset task modes: {sorted(unknown)}")
    return tuple(mode for mode in TASK_MODE_ORDER if mode in task_modes)


@dataclass(frozen=True)
class DatasetSpec:
    """Executable dataset contract kept separate from dataset cards and files."""

    name: str
    dataset_class: type
    params_schema: type[BaseModel] | None
    task_modes: frozenset[TaskMode]
    storage: StorageMode = "file"

    def resolve_location(self, path: str, dataset_id: str | None) -> tuple[str, str]:
        """Resolve the public path/id pair for the common loader constructor."""
        if self.storage == "selector":
            if not dataset_id:
                raise ValueError(f"dataset {self.name!r} requires dataset.id")
            return path, dataset_id
        if dataset_id is not None:
            raise ValueError(f"dataset {self.name!r} does not accept dataset.id; put the full location in dataset.path")
        if self.storage == "directory":
            return path, ""
        expanded = os.path.expanduser(path)
        return os.path.dirname(expanded), os.path.basename(expanded)


class DatasetRegistry:
    """Registry mapping dataset names to dataset classes and schemas."""

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetSpec] = {}

    def register(
        self,
        name: str,
        dataset_cls: type,
        schema: type[BaseModel] | None = None,
        *,
        task_modes: frozenset[TaskMode],
        storage: StorageMode = "file",
    ) -> None:
        """Register a dataset class with an optional parameter schema."""
        if not task_modes:
            raise ValueError(f"dataset {name!r} must support at least one task mode")
        ordered_task_modes(task_modes)
        if storage not in {"file", "directory", "selector"}:
            raise ValueError(f"dataset {name!r} has unknown storage mode {storage!r}")
        spec = DatasetSpec(name, dataset_cls, schema, task_modes, storage)
        existing = self._datasets.get(name)
        if existing is not None and existing != spec:
            raise ValueError(f"dataset {name!r} is already registered differently")
        self._datasets[name] = spec

    def get(self, name: str) -> DatasetSpec:
        """Get one executable dataset specification by name.

        Raises
        ------
        KeyError
            If the dataset is not registered.
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' is not registered")
        return self._datasets[name]

    def names(self) -> list[str]:
        """Return registered dataset names."""
        return sorted(self._datasets.keys())


DATASET_REGISTRY = DatasetRegistry()

DATASET_NAME_MAP = {
    "ETTh1": "tsflab.data.datasets.ett",
    "ETTh2": "tsflab.data.datasets.ett",
    "ETTm1": "tsflab.data.datasets.ett",
    "ETTm2": "tsflab.data.datasets.ett",
    "traffic": "tsflab.data.datasets.custom",
    "weather": "tsflab.data.datasets.custom",
    "electricity": "tsflab.data.datasets.custom",
    # Generic flat-multivariate CSV datasets reuse Dataset_Custom without a
    # per-dataset registry entry; configs set name = "custom".
    "custom": "tsflab.data.datasets.custom",
    "solar": "tsflab.data.datasets.solar",
    "periodic": "tsflab.data.datasets.periodic_data",
    "trend": "tsflab.data.datasets.trend_data",
    "pre_processed": "tsflab.data.datasets.pre_processed",
    "gift_eval": "tsflab.data.datasets.gift_eval",
    # CauAir spatiotemporal / air-quality datasets (index-windowed .npz).
    "cauair_st": "tsflab.data.datasets.cauair",
    "cauair_ts": "tsflab.data.datasets.cauair",
    # Real-time tracks frozen at a release as static datasets.
    "realtime_panel_st": "tsflab.data.datasets.realtime_panel",
    "realtime_panel_ts": "tsflab.data.datasets.realtime_panel",
    # Synthetic node-structured dataset for spatiotemporal-mode smoke tests.
    "synthetic_st": "tsflab.data.datasets.synthetic_st",
}

_REGISTERED_DATASETS: set[str] = set()


def register_dataset_by_name(name: str) -> None:
    """Import and register a dataset using the name map.

    Parameters
    ----------
    name : str
        Dataset name from the config.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the dataset name is not mapped.
    ModuleNotFoundError
        If the mapped module cannot be imported.
    AttributeError
        If the module has no register() function.
    """
    if name in _REGISTERED_DATASETS:
        return
    module_name = DATASET_NAME_MAP.get(name)
    if module_name is None:
        available = ", ".join(sorted(DATASET_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Dataset '{name}' is not mapped. Update DATASET_NAME_MAP in "
            f"tsflab.catalog.registry.datasets. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Dataset registry module not found: {module_name}. Expected module path in DATASET_NAME_MAP"
            ) from exc
        raise ImportError(f"Failed to import '{module_name}' due to missing dependency: {exc}") from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(f"Dataset registry '{module_name}' must define a register() function")
    register_fn()
    _REGISTERED_DATASETS.update(key for key, mapped_module in DATASET_NAME_MAP.items() if mapped_module == module_name)
