"""Summarize resolved run-configuration sweep coverage for `tsf run --dry-run`."""

from __future__ import annotations

import json


def _format_values(values):
    if not values:
        return "-"
    try:
        return ", ".join(str(v) for v in sorted(values))
    except TypeError:
        return ", ".join(str(v) for v in values)


def _flatten_params(params: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in params.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, path))
        elif isinstance(value, (list, tuple)):
            flat[path] = json.dumps(value, ensure_ascii=True)
        else:
            flat[path] = value
    return flat


def _sweep_values(configs, sweep_keys) -> dict:
    sweep_values = {key: set() for key in sweep_keys}
    for cfg in configs:
        flattened = _flatten_params(cfg.raw)
        for key in sweep_keys:
            if key in flattened:
                sweep_values[key].add(flattened[key])
    return sweep_values


def summarize(configs) -> None:
    """Print the resolved sweep coverage of loaded run configurations."""
    print(f"Total runs: {len(configs)}")

    datasets = {cfg.config.dataset.name for cfg in configs}
    models = {cfg.config.model.name for cfg in configs}
    pred_lens = {cfg.config.task.pred_len for cfg in configs}
    seeds = {cfg.config.experiment.random_seed for cfg in configs}

    print(f"Datasets: {_format_values(datasets)}")
    print(f"Models: {_format_values(models)}")
    print(f"Pred lens: {_format_values(pred_lens)}")
    print(f"Seeds: {_format_values(seeds)}")

    sweep_keys = configs[0].sweep_keys if configs else []
    if not sweep_keys:
        return

    sweep_values = _sweep_values(configs, sweep_keys)
    print("Sweep values:")
    for key in sweep_keys:
        print(f"  {key}: {_format_values(sweep_values[key])}")
