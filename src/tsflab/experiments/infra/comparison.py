"""Comparable experiment cohorts with explicit seed and model-variant coverage."""

import math
import statistics
from collections import defaultdict


def _metric_summary(entry, seeds, metric) -> None:
    """Mean, sample std, and count of ``metric`` over seeds that ran exactly once."""
    values = []
    for seed, group in seeds.items():
        if len(group) != 1:
            continue
        try:
            value = float(group[0][metric])
            if math.isfinite(value):
                values.append(value)
        except (KeyError, TypeError, ValueError):
            pass
    entry[metric] = statistics.mean(values) if values else None
    entry[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
    entry[metric + "_n"] = len(values)


def _variant_entry(model, variant, cells, expected) -> dict:
    """Leaderboard entry of one model variant: seed coverage, metrics, and rankability."""
    seeds = defaultdict(list)
    for row in cells:
        seeds[str(row.get("seed"))].append(row)
    duplicate = sorted(seed for seed, group in seeds.items() if len(group) > 1)
    entry = {
        "model": model,
        "variant": variant[:12],
        "seeds": sorted(seeds),
        "missing_seeds": sorted(expected - seeds.keys()),
        "duplicate_seeds": duplicate,
    }
    for metric in ("mse", "mae"):
        _metric_summary(entry, seeds, metric)
    entry["rankable"] = not duplicate and not entry["missing_seeds"] and entry["mse_n"] == len(expected)
    return entry


def _cohort(protocol, records, planned_rows) -> dict:
    """One protocol's ranked leaderboard over its recorded and planned variants."""
    variants = defaultdict(list)
    expected = {str(row.get("seed")) for row in [*records, *planned_rows]}
    for row in planned_rows:
        variants.setdefault((row["model"], row["model_variant"]), [])
    for row in records:
        variants[(row["model"], row["model_variant"])].append(row)
    leaderboard = []
    for (model, variant), cells in variants.items():
        leaderboard.append(_variant_entry(model, variant, cells, expected))
    leaderboard.sort(
        key=lambda item: (
            not item["rankable"],
            item["mse"] if item["mse"] is not None else math.inf,
        )
    )
    return {
        "protocol": protocol,
        "expected_seeds": sorted(expected),
        "leaderboard": leaderboard,
    }


def compare_rows(rows, planned=()):
    """Never merge unknown protocols or different model configurations into a rank."""
    cohorts = defaultdict(list)
    unknown = []
    for row in rows:
        if not row.get("protocol_sha256") or not row.get("model_variant"):
            unknown.append(row)
        else:
            cohorts[row["protocol_sha256"]].append(row)
    planned_cohorts = defaultdict(list)
    for row in planned:
        planned_cohorts[row["protocol_sha256"]].append(row)
        cohorts.setdefault(row["protocol_sha256"], [])
    results = []
    for protocol, records in sorted(cohorts.items()):
        results.append(_cohort(protocol, records, planned_cohorts[protocol]))
    return {"cohorts": results, "unverified_runs": len(unknown)}


def protocol_fingerprint(snapshot, data):
    """One protocol identity shared by result producers and planned-cell reports."""
    # A newly exposed zero cosine floor has the same behavior as historical
    # snapshots that omitted it. Preserve their cohort identity without mutation.
    from copy import deepcopy

    from tsflab.experiments.infra.storage import canonical_hash

    training = deepcopy(snapshot["training"])
    optimizer = training.get("optimizer", {})
    if optimizer.get("min_lr") == 0:
        optimizer.pop("min_lr", None)
    task = deepcopy(snapshot["task"])
    if task.get("window_context") == "carry":
        task.pop("window_context", None)

    return canonical_hash(
        {
            "data": data,
            "task": task,
            "evaluation": snapshot["evaluation"],
            "training": training,
        }
    )
