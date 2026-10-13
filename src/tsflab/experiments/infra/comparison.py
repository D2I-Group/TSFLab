"""Comparable experiment cohorts with explicit seed and model-variant coverage."""

from collections import defaultdict
import math
import statistics


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
        variants = defaultdict(list)
        expected = {
            str(row.get("seed")) for row in [*records, *planned_cohorts[protocol]]
        }
        for row in planned_cohorts[protocol]:
            variants.setdefault((row["model"], row["model_variant"]), [])
        for row in records:
            variants[(row["model"], row["model_variant"])].append(row)
        leaderboard = []
        for (model, variant), cells in variants.items():
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
                entry[metric + "_std"] = (
                    statistics.stdev(values) if len(values) > 1 else None
                )
                entry[metric + "_n"] = len(values)
            entry["rankable"] = (
                not duplicate
                and not entry["missing_seeds"]
                and entry["mse_n"] == len(expected)
            )
            leaderboard.append(entry)
        leaderboard.sort(
            key=lambda item: (
                not item["rankable"],
                item["mse"] if item["mse"] is not None else math.inf,
            )
        )
        results.append(
            {
                "protocol": protocol,
                "expected_seeds": sorted(expected),
                "leaderboard": leaderboard,
            }
        )
    return {"cohorts": results, "unverified_runs": len(unknown)}


def protocol_fingerprint(snapshot, data):
    """One protocol identity shared by result producers and planned-cell reports."""
    from tsflab.experiments.infra.storage import canonical_hash

    # A newly exposed zero cosine floor has the same behavior as historical
    # snapshots that omitted it. Preserve their cohort identity without mutation.
    from copy import deepcopy
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
