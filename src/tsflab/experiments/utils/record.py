"""Emit a self-describing ``record.json`` (a ``tsflab.core.RunRecord``) per run.

Each TSFLab run writes one ``record.json`` next to its CSV outputs. Unlike
``performance.csv`` (positional, shared-header), this is self-describing and
schema-validated, so ``tsf result submit`` and the TSFLab Leaderboard can ingest a run
with zero column-alignment guesswork. Contract or write failures fail the run;
an invalid record is never persisted as if it were a valid experiment artifact.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def build_record_dict(
    *,
    config,
    device,
    run_id: str,
    dataset_id: str,
    metrics: dict,
    fit_time: float | None,
    inference_time: float | None,
    n_windows: int | None = None,
    repo_root: str | None = None,
) -> dict:
    """Assemble + validate a RunRecord, returning its JSON-able dict.

    Raises on validation failure; callers should catch and fall back.
    """
    from tsflab.core import (
        HorizonResult,
        MetricSet,
        RunConfigRef,
        RunEnv,
        RunRecord,
        RunTiming,
    )
    from tsflab.experiments.infra.storage import canonical_hash
    from tsflab.experiments.utils.env import collect_env, collect_git

    snapshot = config.model_dump(mode="json")
    mode = config.task.mode
    # Track defaults to the task mode, but a dataset config may override it
    # (e.g. track = "realtime" for live datasets like stock_hs300).
    track = getattr(config.dataset, "track", None) or mode
    metric_keys = set(MetricSet.model_fields)
    metric_set = MetricSet(**{k: v for k, v in metrics.items() if k in metric_keys})

    horizon = HorizonResult(
        horizon=config.task.pred_len,
        metrics=metric_set,
        timing=RunTiming(fit_time_sec=fit_time, inference_time_sec=inference_time),
        n_windows=n_windows,
        run_id=run_id,
    )

    env_fields = set(RunEnv.model_fields)
    env_data = collect_env(device)
    env_data.update(collect_git(repo_root))
    env = RunEnv(**{k: v for k, v in env_data.items() if k in env_fields})

    record = RunRecord(
        record_id=(f"{config.model.name}__{dataset_id}__seed{config.experiment.random_seed}__pl{config.task.pred_len}"),
        model=config.model.name,
        dataset_id=dataset_id,
        track=track,
        mode=mode,
        seed=config.experiment.random_seed,
        results=[horizon],
        config=RunConfigRef(
            mode=mode,
            seq_len=config.task.seq_len,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            features=config.task.features,
            config_sha256=canonical_hash(snapshot),
            snapshot=snapshot,
        ),
        env=env,
        created_at=_iso_now(),
    )
    return record.model_dump(mode="json")


def write_run_record(record_path: str, **kwargs) -> None:
    """Build and write one schema-valid ``record.json`` or raise."""
    data = build_record_dict(**kwargs)
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
