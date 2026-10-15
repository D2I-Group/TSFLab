"""Validate submissions and aggregate them into a ranked leaderboard.

This is the single implementation used by ``tsf result leaderboard`` and by the
site pipeline (``apps/web/pipeline``). It depends only on pydantic and the
standard library. Two submission shapes are accepted:

* a flat :class:`RunRecord` (``{model, dataset_id, track, results: [...]}``);
* a :class:`SubmissionReport` bundle (``{manifest, datasets, records: [...]}``).

Rows average every run of the same (track, dataset, horizon, model) and record
the run count, standard deviations, and the submission ids behind them, so each
row can be traced to its evidence.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from .run_record import RunRecord
from .submission import SubmissionReport

PRIMARY_METRIC = "mse"
METRIC_KEYS = ("mse", "mae", "rmse", "corr", "wape", "rse")


def iter_records(doc: dict) -> Iterator[dict]:
    """Yield flat records from either submission shape."""
    if isinstance(doc.get("records"), list):
        yield from doc["records"]
    elif "model" in doc and "results" in doc:
        yield doc


def validate_submission(doc: dict) -> list[str]:
    """Return contract violations for one ``submission.json`` document."""
    errors: list[str] = []
    try:
        (SubmissionReport if isinstance(doc.get("records"), list) else RunRecord).model_validate(doc)
    except ValidationError as exc:
        errors.extend(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    bound = (
        any(ds.get("source_config") for ds in doc.get("datasets", []) or [])
        or bool((doc.get("env") or {}).get("git_sha"))
        or bool(doc.get("config"))
        or any((r.get("env") or {}).get("git_sha") or r.get("config") for r in doc.get("records", []) or [])
    )
    if not bound:
        errors.append("TSFLab binding missing: need datasets[].source_config, env.git_sha, or config")
    return errors


def load_submissions(root: Path) -> tuple[list[tuple[Path, dict]], dict[Path, list[str]]]:
    """Read every ``submission.json`` under ``root``; return (valid, rejected)."""
    valid, rejected = [], {}
    for path in sorted(Path(root).rglob("submission.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected[path] = [f"unreadable: {exc}"]
            continue
        errors = validate_submission(doc)
        if errors:
            rejected[path] = errors
        else:
            valid.append((path, doc))
    return valid, rejected


def _mean_std(values: list) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return round(mean, 6), None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return round(mean, 6), round(math.sqrt(var), 6)


def aggregate(docs: Iterable[dict], primary: str = PRIMARY_METRIC) -> dict:
    """Aggregate records into ``{track: {"datasets": {id: {"horizons": {h: rows}}}}}``.

    Keys are the canonical ``track`` and ``dataset_id`` of each record; display
    names are the presentation layer's concern.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for doc in docs:
        manifest_id = (doc.get("manifest") or {}).get("submission_id")
        for rec in iter_records(doc):
            model, dataset = rec.get("model"), rec.get("dataset_id")
            if not model or not dataset:
                continue
            sid = rec.get("record_id") or manifest_id or model
            for result in rec.get("results", []):
                metrics = result.get("metrics") or {}
                groups[(rec.get("track") or "time_series", dataset, str(result.get("horizon")), model)].append(
                    {**{k: metrics.get(k) for k in METRIC_KEYS}, "_sid": sid}
                )
    tracks: dict = {}
    for (track, dataset, horizon, model), runs in groups.items():
        row: dict = {"model": model}
        for key in METRIC_KEYS:
            mean, std = _mean_std([r[key] for r in runs])
            row[key] = mean
            if std is not None:
                row[f"{key}_std"] = std
        row["n_runs"] = len(runs)
        row["submission_ids"] = sorted({r["_sid"] for r in runs})
        (
            tracks.setdefault(track, {"datasets": {}})["datasets"]
            .setdefault(dataset, {"horizons": {}})["horizons"]
            .setdefault(horizon, [])
        ).append(row)
    for block in tracks.values():
        for dataset in block["datasets"].values():
            horizons = dataset["horizons"]
            for rows in horizons.values():
                rows.sort(key=lambda r: (r.get(primary) is None, r.get(primary)))
                for rank, row in enumerate(rows, start=1):
                    row["rank"] = rank
            dataset["horizons"] = {
                h: horizons[h] for h in sorted(horizons, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
            }
    return tracks
