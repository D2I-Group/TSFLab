"""Load and expand benchmark configs with extends and sweep support."""

from __future__ import annotations

import os
from contextvars import ContextVar

_EXTENDS_STACK = ContextVar("extends_stack", default=())
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import tomllib

from tsflab.experiments.config.schema.root import RootConfig
from tsflab.catalog.registry.datasets import DATASET_REGISTRY, register_dataset_by_name
from tsflab.catalog.registry.models import MODEL_CATALOG

PACKAGE_SCHEME = "tsflab://"


def validate_task_compatibility(mode: str, dataset_spec, model_spec) -> None:
    """Reject a run whose data layout is unsupported by either endpoint."""
    if mode not in dataset_spec.task_modes:
        raise ValueError(
            f"dataset {dataset_spec.name!r} does not support task.mode={mode!r}; "
            f"supported: {sorted(dataset_spec.task_modes)}"
        )
    if mode not in model_spec.task_modes:
        raise ValueError(
            f"model {model_spec.name!r} does not support task.mode={mode!r}; "
            f"supported: {sorted(model_spec.task_modes)}"
        )


@dataclass(frozen=True)
class LoadedConfig:
    """Container for validated and expanded configs.

    Parameters
    ----------
    raw : dict
        Expanded config dictionary (post-extends and sweep).
    config : RootConfig
        Validated config object.
    sweep_keys : list[str]
        Dot-delimited keys used for sweep expansion.
    config_name : str
        Stem name of the root config file.
    """

    raw: dict
    config: RootConfig
    sweep_keys: list[str]
    config_name: str


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge dictionaries, replacing lists and scalars.

    Parameters
    ----------
    base : dict[str, Any]
        Base dictionary to merge into.
    override : dict[str, Any]
        Values to apply on top of the base.

    Returns
    -------
    dict[str, Any]
        New dictionary with merged content.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            result[key] = deepcopy(value)
            continue

        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_toml(path: str) -> dict:
    """Read a TOML file into a dict.

    Parameters
    ----------
    path : str
        TOML file path.

    Returns
    -------
    dict
        Parsed TOML content.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_extends_list(extends: str | Iterable[str], base_dir: str) -> dict:
    """Resolve a list of extends directives into a merged config.

    Parameters
    ----------
    extends : str | Iterable[str]
        Extends directive(s) to resolve.
    base_dir : str
        Base directory to resolve relative extends paths.

    Returns
    -------
    dict
        Merged config from the extends chain.
    """
    if isinstance(extends, str):
        extends_list = [extends]
    else:
        extends_list = list(extends)

    merged: dict[str, Any] = {}
    for rel_path in extends_list:
        file_path = rel_path
        if rel_path.startswith(PACKAGE_SCHEME):
            # Resolve against the checkout or the installed package's assets,
            # so standalone projects inherit catalog configs without copying.
            from tsflab.core.paths import repository_root

            file_path = str(repository_root() / rel_path[len(PACKAGE_SCHEME):])
        elif not os.path.isabs(rel_path):
            file_path = os.path.join(base_dir, rel_path)
        absolute = os.path.realpath(file_path)
        stack = _EXTENDS_STACK.get()
        if absolute in stack:
            raise ValueError("cyclic config extends: " + " -> ".join((*stack, absolute)))
        token = _EXTENDS_STACK.set((*stack, absolute))
        try:
            base_cfg = _read_toml(file_path)
            base_cfg = _resolve_extends(base_cfg, os.path.dirname(file_path))
        finally:
            _EXTENDS_STACK.reset(token)
        merged = deep_merge(merged, base_cfg)
    return merged


def _resolve_extends(cfg: dict, base_dir: str) -> dict:
    """Resolve and merge any extends directives in a config.

    Parameters
    ----------
    cfg : dict
        Raw config dictionary.
    base_dir : str
        Base directory to resolve relative extends paths.

    Returns
    -------
    dict
        Merged config with extends resolved.
    """
    if "extends" not in cfg:
        return cfg

    merged = _resolve_extends_list(cfg["extends"], base_dir)

    cfg = {k: v for k, v in cfg.items() if k != "extends"}
    return deep_merge(merged, cfg)


def _expand_sweep_extends(
    extends_cfg: dict,
    run_cfg: dict,
    base_dir: str,
) -> tuple[list[dict], list[str]]:
    """Expand sweep.extend tables into a list of base configs.

    Parameters
    ----------
    extends_cfg : dict
        Config merged from the extends chain.
    run_cfg : dict
        Config values from the root run file (without extends).
    base_dir : str
        Base directory to resolve relative extend paths.

    Returns
    -------
    tuple[list[dict], list[str]]
        Expanded base configs and sweep key labels for extend axes.
    """
    sweep = run_cfg.get("sweep")
    if not sweep or not isinstance(sweep, dict) or "extend" not in sweep:
        merged = deep_merge(extends_cfg, run_cfg)
        return [merged], []

    extend_axes = sweep.get("extend")
    if not isinstance(extend_axes, dict):
        raise ValueError("sweep.extend must be a table of axis names to paths")

    axis_names = list(extend_axes.keys())
    axis_values: list[list[Any]] = []
    for axis in axis_names:
        values = extend_axes[axis]
        if isinstance(values, list):
            axis_values.append(list(values))
        else:
            axis_values.append([values])

    def _product(items: list[list[Any]], prefix: list[Any] | None = None):
        prefix = prefix or []
        if not items:
            yield prefix
            return
        for item in items[0]:
            yield from _product(items[1:], prefix + [item])

    run_cfg_no_extend = deepcopy(run_cfg)
    sweep_no_extend = deepcopy(sweep)
    sweep_no_extend.pop("extend", None)
    if sweep_no_extend:
        run_cfg_no_extend["sweep"] = sweep_no_extend
    else:
        run_cfg_no_extend.pop("sweep", None)

    base_configs: list[dict] = []
    extend_keys = [f"extend.{axis}" for axis in axis_names]
    for combo in _product(axis_values):
        combo_cfg: dict[str, Any] = {}
        extend_meta: dict[str, Any] = {}
        for axis, rel_path in zip(axis_names, combo):
            file_path = rel_path
            if not os.path.isabs(rel_path):
                file_path = os.path.join(base_dir, rel_path)
            axis_cfg = _read_toml(file_path)
            axis_cfg = _resolve_extends(axis_cfg, os.path.dirname(file_path))
            combo_cfg = deep_merge(combo_cfg, axis_cfg)
            extend_meta[axis] = os.path.splitext(os.path.basename(file_path))[0]

        merged = deep_merge(extends_cfg, combo_cfg)
        merged = deep_merge(merged, run_cfg_no_extend)
        if extend_meta:
            merged = deep_merge(merged, {"extend": extend_meta})
        base_configs.append(merged)

    return base_configs, extend_keys


def _set_by_path(obj: dict, path: str, value: Any) -> None:
    """Set a nested dict value using a dot-delimited path.

    Parameters
    ----------
    obj : dict
        Object to update in place.
    path : str
        Dot-delimited path.
    value : Any
        Value to set at the target path.

    Returns
    -------
    None
    """
    keys = path.split(".")
    current = obj
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _flatten_sweep(sweep: dict, prefix: str = "") -> dict:
    """Flatten nested sweep keys into dot-delimited paths.

    Parameters
    ----------
    sweep : dict
        Sweep definition with optional nested dictionaries.
    prefix : str, optional
        Prefix used during recursion.

    Returns
    -------
    dict
        Flat mapping of dot-delimited keys to values.
    """
    flat = {}
    for key, value in sweep.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_sweep(value, path))
        else:
            flat[path] = value
    return flat


def _expand_sweep(base_cfg: dict) -> Iterable[dict]:
    """Yield configs expanded across all sweep combinations.

    Parameters
    ----------
    base_cfg : dict
        Raw config dictionary containing an optional sweep section.

    Yields
    ------
    dict
        Expanded config dictionaries without the sweep section.
    """
    sweep = base_cfg.get("sweep")
    if not sweep:
        yield {k: v for k, v in base_cfg.items() if k != "sweep"}
        return

    sweep = _flatten_sweep(sweep)
    keys = list(sweep.keys())
    values_list = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]

    def _product(items: list[list[Any]], prefix: list[Any] | None = None):
        prefix = prefix or []
        if not items:
            yield prefix
            return
        for item in items[0]:
            yield from _product(items[1:], prefix + [item])

    base_cfg = {k: v for k, v in base_cfg.items() if k != "sweep"}
    for combo in _product(values_list):
        override = {}
        for key, value in zip(keys, combo):
            _set_by_path(override, key, value)
        yield deep_merge(base_cfg, override)


def load_config(path: str) -> list[LoadedConfig]:
    """Load a TOML config file into validated config objects.

    Parameters
    ----------
    path : str
        Path to a root TOML config file.

    Returns
    -------
    list[LoadedConfig]
        Expanded and validated configs for a sweep.
    """
    return load_mapping(_read_toml(path), base_dir=os.path.dirname(path),
                        config_name=os.path.splitext(os.path.basename(path))[0])


def load_mapping(raw_cfg: dict, *, base_dir: str = ".", config_name: str = "request") -> list[LoadedConfig]:
    """Validate and expand a declarative request through the same TOML loader."""
    raw_cfg = deepcopy(raw_cfg)
    extends_cfg: dict[str, Any] = {}
    if "extends" in raw_cfg:
        extends_cfg = _resolve_extends_list(raw_cfg["extends"], base_dir)

    run_cfg = {k: v for k, v in raw_cfg.items() if k != "extends"}
    base_configs, extend_keys = _expand_sweep_extends(extends_cfg, run_cfg, base_dir)

    configs: list[LoadedConfig] = []
    for base_cfg in base_configs:
        sweep_keys: list[str] = []
        sweep = base_cfg.get("sweep")
        if sweep:
            sweep_keys = list(_flatten_sweep(sweep).keys())
        if extend_keys:
            sweep_keys = extend_keys + sweep_keys

        for expanded in _expand_sweep(base_cfg):
            # "extend" is metadata `_expand_sweep_extends` injects to label
            # which axis file produced this combo (consumed via `raw` below,
            # e.g. by `tsf run --dry-run`) — not a RootConfig section, so
            # it's excluded from validation rather than left for pydantic to
            # silently ignore.
            config = RootConfig.model_validate(
                {k: v for k, v in expanded.items() if k != "extend"}
            )
            register_dataset_by_name(config.dataset.name)
            dataset_spec = DATASET_REGISTRY.get(config.dataset.name)
            if dataset_spec.params_schema is not None:
                config.dataset.params = dataset_spec.params_schema.model_validate(
                    config.dataset.params
                )
            model_spec = MODEL_CATALOG.get(config.model.name)
            validate_task_compatibility(config.task.mode, dataset_spec, model_spec)
            config.model.params = model_spec.validate_params(config.model.params)
            configs.append(
                LoadedConfig(
                    raw=expanded,
                    config=config,
                    sweep_keys=sweep_keys,
                    config_name=config_name,
                )
            )
    return configs


def load_model_request(name: str, settings=(), axes=()) -> list[LoadedConfig]:
    """Resolve a named preset plus dotted KEY=JSON overrides without writing TOML.

    Scalar strings may be unquoted. Every value is validated before dispatch;
    expanded snapshots retain all supplied scientific settings and sweep axes.
    """
    import json
    spec = MODEL_CATALOG.get(name)
    raw = {"extends": ["tsflab://configs/base.toml", f"tsflab://{spec.config_path}"]}
    for entries, sweep in ((settings, False), (axes, True)):
        seen = set()
        for entry in entries:
            key, separator, text = entry.partition("=")
            key = key.strip()
            if not separator or len(key.split(".")) < 2 or any(not part for part in key.split(".")):
                raise ValueError("expected dotted KEY=VALUE, e.g. task.seq_len=96")
            if key in seen:
                raise ValueError(f"duplicate request key: {key}")
            seen.add(key)
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            if sweep and (not isinstance(value, list) or not value):
                raise ValueError(f"sweep {key} requires a nonempty JSON list")
            if key.split(".")[0] not in RootConfig.model_fields:
                raise ValueError(f"unknown request section: {key.split('.')[0]}")
            _set_by_path(raw, "sweep." + key if sweep else key, value)
    return load_mapping(raw, config_name=f"request-{name}")
