"""Replay the rolling protocol over historical weeks.

Each replayed round is opened at a past weekly cutoff on a store that already
contains the following weeks, forecast by baselines (and optionally catalog
models trained only on data up to that cutoff), and scored immediately. The
rounds are written to a separate root so they never mix with live rounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tsflab.realtime import rounds as R
from tsflab.realtime.baselines import run_baselines
from tsflab.realtime.store import PanelStore
from tsflab.realtime.tracks import TrackSpec

REPLAY_ROOT = Path("work_dirs") / "_realtime_replay"


def weekly_cutoffs(store: PanelStore, track: TrackSpec, end: pd.Timestamp, weeks: int) -> list[pd.Timestamp]:
    """Last observed timestamp before each of the ``weeks`` Mondays ending at ``end``."""
    index = store.read(start=end - pd.Timedelta(weeks=weeks + 2), end=end).dropna(how="all").index
    mondays = pd.date_range(end=end, periods=weeks, freq="W-MON")
    return [index[index < monday].max() for monday in mondays if (index < monday).any()]


def replay(
    track: TrackSpec,
    store: PanelStore,
    *,
    end: pd.Timestamp,
    weeks: int,
    models: list[str] | None = None,
    root: Path = REPLAY_ROOT,
    work_dir: Path = Path("work_dirs") / "_realtime_replay_runs",
    overrides: dict | None = None,
) -> dict:
    for cutoff in weekly_cutoffs(store, track, end, weeks):
        opened = (cutoff + pd.Timedelta(hours=1)).tz_localize(track.tz)
        spec = R.open_round(track, store, now=opened.to_pydatetime(), root=root, cutoff=cutoff)
        existing = {p.stem for p in (R.round_dir(track.id, spec.round_id, root) / "forecasts").glob("*.json")}
        for submission in run_baselines(spec, store, track):
            if submission.model not in existing:
                R.write_forecast(submission.model_copy(update={"submitted_at": spec.opened_at}), spec, root)
        for model in models or []:
            if model not in existing:
                _replay_model(spec, store, track, model, work_dir, overrides, root)
        scores = R.score_round(spec, store, track.min_coverage, root)
        print(
            f"{track.id} {spec.round_id}: cutoff {spec.cutoff} "
            f"{'scored ' + str(len(scores)) if scores else 'incomplete truth'}"
        )
    return R.track_summary(track.id, root)


def _replay_model(spec, store: PanelStore, track: TrackSpec, model: str, work_dir: Path, overrides, root: Path) -> None:
    from tsflab.realtime.forecast import forecast_with_model

    try:
        submission = forecast_with_model(spec, store, track, model, work_dir, json.loads(json.dumps(overrides or {})))
    except Exception as exc:  # one failing model must not stop the replay
        print(f"{track.id} {spec.round_id}: {model} failed: {exc}")
        return
    R.write_forecast(submission.model_copy(update={"submitted_at": spec.opened_at}), spec, root)
