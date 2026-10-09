"""Helpers for writing CSV summaries and sweep metadata."""

from __future__ import annotations

import csv
import os
import json
from typing import Iterable


def _read_existing_header(path: str) -> list[str]:
    """Return the header row already written to a CSV file (``[]`` if empty)."""
    with open(path, "r", newline="") as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            return []


def write_csv_summary(path: str, row: dict, header: Iterable[str] | None = None) -> None:
    """Atomically upsert a run row; parallel writers and retries cannot duplicate it."""
    from pathlib import Path
    from tsflab.experiments.infra.storage import file_lock
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(destination.with_suffix(destination.suffix + ".lock")):
        records, existing = [], []
        if destination.exists():
            with destination.open(newline="") as stream:
                reader = csv.DictReader(stream)
                existing = reader.fieldnames or []
                records = list(reader)
        fields = list(header) if header is not None else list(dict.fromkeys([*existing, *row]))
        if row.get("run_id"):
            records = [old for old in records if old.get("run_id") != str(row["run_id"])]
        records.append(row)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)


def _flatten_params(params: dict, prefix: str = "") -> dict:
    """Flatten nested params into dot-delimited keys.

    Lists/tuples are JSON-encoded to preserve structure in CSV outputs.

    Parameters
    ----------
    params : dict
        Parameter dictionary.
    prefix : str, optional
        Prefix used during recursion.

    Returns
    -------
    dict
        Flattened parameter mapping.
    """
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


def _append_sweep_values(row: dict, raw: dict, sweep_keys: list[str]) -> None:
    """Append sweep values to the output row in place.

    Parameters
    ----------
    row : dict
        Summary row to update.
    raw : dict
        Raw expanded config dictionary.
    sweep_keys : list[str]
        Dot-delimited sweep keys to include.

    Returns
    -------
    None
    """
    if not sweep_keys:
        return
    flattened = _flatten_params(raw)
    for key in sweep_keys:
        if key in flattened:
            row[f"sweep.{key}"] = flattened[key]


def default_summary_row(
    base: dict,
    metrics: dict[str, float],
    raw: dict | None = None,
    sweep_keys: list[str] | None = None,
) -> dict:
    """Build a normalized summary row for CSV output.

    Parameters
    ----------
    base : dict
        Required metadata fields (dataset, model, lengths, seed, run_id).
    metrics : dict[str, float]
        Metric values to include.
    raw : dict | None, optional
        Raw expanded config for sweep metadata.
    sweep_keys : list[str] | None, optional
        Dot-delimited sweep keys to include.

    Returns
    -------
    dict
        Output row dictionary.
    """
    row = {
        "dataset": base.get("dataset"),
        "model": base.get("model"),
        "seq_len": base.get("seq_len"),
        "pred_len": base.get("pred_len"),
        "seed": base.get("seed"),
        "run_id": base.get("run_id"),
    }

    metric_order = ["mae", "mse", "rmse", "mape", "mspe"]
    for name in metric_order:
        if name in metrics:
            row[name] = metrics[name]
    for name, value in metrics.items():
        if name not in row:
            row[name] = value

    # Timing columns. ``fit_time`` is the total training wall-clock; the
    # ``inference_time`` is the test-set evaluation wall-clock. Only emitted when
    # provided so existing callers/headers are unaffected.
    if "fit_time" in base:
        row["fit_time"] = base["fit_time"]
    if "inference_time" in base:
        row["inference_time"] = base["inference_time"]
    # Cost columns of a run: trainable parameters, epochs trained, peak training GPU memory (MB).
    for name in ("params", "epochs", "peak_train_mem_mb"):
        if base.get(name) is not None:
            row[name] = base[name]

    if raw and sweep_keys:
        _append_sweep_values(row, raw, sweep_keys)
    return row
