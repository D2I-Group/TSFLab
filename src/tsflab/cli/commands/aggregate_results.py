"""Aggregate performance and profile CSVs for a dataset."""

from __future__ import annotations

import argparse

from tsflab.cli.args import add_name, add_option, resolve_name
import csv
import glob
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FilterCondition:
    key: str
    op: str
    value: str


DEFAULT_PERF_FIELDS = "model,seq_len,pred_len,mse,mae"
DEFAULT_PROF_FIELDS = "latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb"


def _parse_filters(filter_expr: str | None) -> list[FilterCondition]:
    if not filter_expr:
        return []
    conditions: list[FilterCondition] = []
    for raw in filter_expr.split(","):
        token = raw.strip()
        if not token:
            continue
        op = None
        for candidate in ("<=", ">=", "!=", "=", "<", ">", "~"):
            if candidate in token:
                op = candidate
                break
        if op is None:
            raise ValueError(f"Invalid filter token: {token}")
        key, value = token.split(op, 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid filter token: {token}")
        conditions.append(FilterCondition(key=key, op=op, value=value))
    return conditions


def _to_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_condition(row: dict[str, str], condition: FilterCondition) -> bool:
    if condition.key not in row:
        return False
    raw_value = row.get(condition.key, "")
    if condition.op == "~":
        return condition.value in raw_value

    left_num = _to_number(raw_value)
    right_num = _to_number(condition.value)
    if left_num is not None and right_num is not None:
        if condition.op == "=":
            return left_num == right_num
        if condition.op == "!=":
            return left_num != right_num
        if condition.op == "<":
            return left_num < right_num
        if condition.op == ">":
            return left_num > right_num
        if condition.op == "<=":
            return left_num <= right_num
        if condition.op == ">=":
            return left_num >= right_num
        return False

    if condition.op == "=":
        return raw_value == condition.value
    if condition.op == "!=":
        return raw_value != condition.value
    if condition.op == "<":
        return raw_value < condition.value
    if condition.op == ">":
        return raw_value > condition.value
    if condition.op == "<=":
        return raw_value <= condition.value
    if condition.op == ">=":
        return raw_value >= condition.value
    return False


def _filter_rows(
    rows: Iterable[dict[str, str]], conditions: list[FilterCondition]
) -> list[dict[str, str]]:
    if not conditions:
        return list(rows)
    result = []
    for row in rows:
        if all(_match_condition(row, cond) for cond in conditions):
            result.append(row)
    return result


def _collect_csv_files(dataset_dir: str, filename: str) -> list[str]:
    pattern = os.path.join(dataset_dir, "*", filename)
    return sorted(glob.glob(pattern))


def _read_csvs(paths: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    return fieldnames, rows


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_fields(fields_expr: str | None, fieldnames: list[str]) -> list[str]:
    if fields_expr is None:
        return fieldnames
    tokens = [token.strip() for token in fields_expr.split(",") if token.strip()]
    if not tokens:
        return fieldnames
    selected = [name for name in tokens if name in fieldnames]
    missing = [name for name in tokens if name not in fieldnames]
    if missing:
        print(f"Warning: missing fields ignored: {', '.join(missing)}")
    return selected


def _aggregate_values(values: list[float], how: str) -> float:
    if how == "mean":
        return sum(values) / len(values)
    if how == "median":
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        if n % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0
    if how == "max":
        return max(values)
    raise ValueError(f"Unknown aggregate: {how}")


def _collapse_rows(
    rows: list[dict[str, str]],
    metric_cols: list[str],
    how: str,
    null_threshold: float | None,
) -> tuple[list[dict[str, str]], list[str]]:
    """TFB-style fairness collapse for the aggregate tool.

    Collapses to one row per (model, pred_len) cell, aggregating each metric column
    across seeds/runs with ``how``. A cell is "missing" for a model if it has no
    non-numeric/NaN value for any requested metric. If ``null_threshold`` is set,
    models whose missing-cell fraction exceeds it are dropped (and logged).
    Remaining NaN metric cells are filled with that metric's column mean.
    Returns (collapsed_rows, dropped_log_lines).
    """
    log: list[str] = []
    # All pred_len cells observed (the denominator for null fractions).
    all_cells = sorted({r.get("pred_len", "") for r in rows})
    n_cells = len(all_cells)

    # Group numeric metric values by (model, pred_len).
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    models: list[str] = []
    for row in rows:
        model = row.get("model", "")
        pred_len = row.get("pred_len", "")
        if model not in models:
            models.append(model)
        cell = (model, pred_len)
        bucket = by_cell.setdefault(cell, {m: [] for m in metric_cols})
        for m in metric_cols:
            num = _to_number(row.get(m, ""))
            if num is not None:
                bucket[m].append(num)

    # Per-model missing-cell count: a (model, pred_len) cell is missing if any metric
    # has no numeric value there (covers absent rows and NaN metric values alike).
    missing_by_model: dict[str, int] = {m: 0 for m in models}
    for model in models:
        for pred_len in all_cells:
            bucket = by_cell.get((model, pred_len))
            if bucket is None or any(not bucket[m] for m in metric_cols):
                missing_by_model[model] += 1

    keep_models = set(models)
    if null_threshold is not None and n_cells > 0:
        dropped = []
        for model in models:
            frac = missing_by_model[model] / float(n_cells)
            if frac > null_threshold:
                dropped.append((model, frac))
        if dropped:
            log.append(
                f"[fairness] null-threshold={null_threshold}: excluding "
                f"{len(dropped)} model(s) of {len(models)} ({n_cells} cells each):"
            )
            for model, frac in sorted(dropped, key=lambda kv: -kv[1]):
                missing_n = round(frac * n_cells)
                log.append(
                    f"  - {model}: {missing_n}/{n_cells} cells missing "
                    f"(null_frac={frac:.3f} > {null_threshold})"
                )
                keep_models.discard(model)
        else:
            log.append(
                f"[fairness] null-threshold={null_threshold}: no models exceed "
                f"threshold; none excluded."
            )

    # Build collapsed rows for surviving (model, pred_len) cells.
    collapsed: list[dict[str, str]] = []
    cell_values: dict[tuple[str, str], dict[str, float | None]] = {}
    for (model, pred_len), bucket in by_cell.items():
        if model not in keep_models:
            continue
        agg: dict[str, float | None] = {}
        for m in metric_cols:
            agg[m] = _aggregate_values(bucket[m], how) if bucket[m] else None
        cell_values[(model, pred_len)] = agg

    # Fill remaining NaNs (cells with no numeric value) with the metric column mean.
    for m in metric_cols:
        present = [v[m] for v in cell_values.values() if v[m] is not None]
        col_mean = sum(present) / len(present) if present else None
        n_missing = sum(1 for v in cell_values.values() if v[m] is None)
        if n_missing and col_mean is not None:
            log.append(
                f"[fairness] fill-nan-with-mean: filling {n_missing} NaN {m} "
                f"cell(s) with column mean={col_mean:.6f}"
            )
            for v in cell_values.values():
                if v[m] is None:
                    v[m] = col_mean

    for (model, pred_len), agg in sorted(cell_values.items()):
        out: dict[str, str] = {"model": model, "pred_len": pred_len}
        for m in metric_cols:
            out[m] = "" if agg[m] is None else repr(agg[m])
        collapsed.append(out)

    return collapsed, log


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tsf result aggregate",
        description="Aggregate performance and profile CSVs for a dataset"
    )
    add_name(parser, "dataset", flag="--dataset", help="dataset name (work_dirs/<dataset>/)")
    add_option(parser, "--records", "work_dir", "--work-dir", type=str, default="./work_dirs",
               help="run-records root (default ./work_dirs)")
    add_option(parser, "--out", "output", "--output", type=str, default=None,
               help="output CSV path (default: <records>/<dataset>/results_all.csv)")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter expression, e.g. 'pred_len=96,model~Linear'",
    )
    parser.add_argument(
        "--perf-fields",
        type=str,
        default=DEFAULT_PERF_FIELDS,
        help="Comma-separated fields to keep from performance.csv",
    )
    parser.add_argument(
        "--prof-fields",
        type=str,
        default=DEFAULT_PROF_FIELDS,
        help="Comma-separated fields to keep from profile.csv",
    )
    # --- TFB-style fairness policy (opt-in; defaults preserve prior behavior) ---
    parser.add_argument(
        "--collapse",
        action="store_true",
        help=(
            "TFB fairness: collapse rows to one per (model, pred_len) by aggregating "
            "the metric columns across seeds with --aggregate, instead of emitting "
            "raw per-run rows. Off by default (raw passthrough preserved)."
        ),
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median", "max"],
        default="mean",
        help=(
            "TFB fairness: aggregation used by --collapse to combine a metric across "
            "seeds/runs within a (model, pred_len) cell (default: mean). Ignored "
            "unless --collapse is set."
        ),
    )
    parser.add_argument(
        "--null-threshold",
        type=float,
        default=None,
        help=(
            "TFB fairness: with --collapse, exclude any model that is NaN/missing on "
            "more than this fraction of the (pred_len) cells. Unset (default) "
            "disables exclusion. Dropped models are logged, not silently truncated. "
            "Typical value: 0.3."
        ),
    )
    parser.add_argument(
        "--metric-cols",
        type=str,
        default="mse,mae",
        help=(
            "Comma-separated metric columns the fairness policy aggregates and checks "
            "for nulls (default: mse,mae). Only used with --collapse."
        ),
    )
    args = parser.parse_args()
    resolve_name(parser, args, "dataset")

    dataset_dir = os.path.join(args.work_dir, args.dataset)
    perf_paths = _collect_csv_files(dataset_dir, "performance.csv")
    prof_paths = _collect_csv_files(dataset_dir, "profile.csv")

    if not perf_paths and not prof_paths:
        raise SystemExit(f"No performance.csv or profile.csv found under {dataset_dir}")

    perf_fieldnames: list[str] = []
    perf_rows: list[dict[str, str]] = []
    if perf_paths:
        perf_fieldnames, perf_rows = _read_csvs(perf_paths)

    prof_fieldnames: list[str] = []
    prof_rows: list[dict[str, str]] = []
    if prof_paths:
        prof_fieldnames, prof_rows = _read_csvs(prof_paths)

    perf_fields = _parse_fields(args.perf_fields, perf_fieldnames)
    prof_fields = _parse_fields(args.prof_fields, prof_fieldnames)
    if perf_paths and not perf_fields:
        raise SystemExit("No valid performance fields selected for output")
    if prof_paths and not prof_fields:
        raise SystemExit("No valid profile fields selected for output")

    prof_by_run: dict[str, dict[str, str]] = {}
    if prof_rows:
        for row in prof_rows:
            run_id = row.get("run_id", "")
            if run_id:
                prof_by_run[run_id] = row

    merged_rows: list[dict[str, str]] = []
    if perf_rows:
        for row in perf_rows:
            merged: dict[str, str] = {}
            for name in perf_fields:
                merged[name] = row.get(name, "")
            if prof_rows:
                prof_row = prof_by_run.get(row.get("run_id", ""))
                if prof_row:
                    for name in prof_fields:
                        merged[name] = prof_row.get(name, "")
                else:
                    for name in prof_fields:
                        merged[name] = ""
            merged_rows.append(merged)
    else:
        for row in prof_rows:
            merged: dict[str, str] = {}
            for name in prof_fields:
                merged[name] = row.get(name, "")
            merged_rows.append(merged)

    conditions = _parse_filters(args.filter)
    filtered_rows = _filter_rows(merged_rows, conditions)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(dataset_dir, "results_all.csv")
    output_fields = perf_fields + prof_fields if perf_rows else prof_fields

    output_rows = filtered_rows
    if args.collapse:
        metric_cols = [
            c.strip() for c in args.metric_cols.split(",") if c.strip()
        ]
        missing_metrics = [c for c in metric_cols if c not in output_fields]
        if missing_metrics:
            print(
                "Warning: metric columns not present, fairness collapse may be "
                f"empty: {', '.join(missing_metrics)}"
            )
        output_rows, fairness_log = _collapse_rows(
            filtered_rows, metric_cols, args.aggregate, args.null_threshold
        )
        for line in fairness_log:
            print(line)
        # Collapsed rows only carry model, pred_len, and metric columns.
        output_fields = ["model", "pred_len"] + metric_cols

    _write_csv(output_path, output_fields, output_rows)

    print(f"Performance files: {len(perf_paths)} | Profile files: {len(prof_paths)}")
    if args.collapse:
        print(
            f"Aggregated {len(merged_rows)} rows; kept {len(filtered_rows)} after "
            f"filter; collapsed to {len(output_rows)} (model, pred_len) rows "
            f"via --aggregate {args.aggregate}."
        )
    else:
        print(f"Aggregated {len(merged_rows)} rows; kept {len(filtered_rows)} rows.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
