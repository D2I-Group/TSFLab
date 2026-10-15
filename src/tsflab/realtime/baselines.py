"""Reference forecasters that run on CPU in seconds, so every round is populated.

They are deliberately simple anchors: a method that cannot beat the seasonal
naive forecast on a round has not learned anything the calendar did not know.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from tsflab.core.realtime import ForecastSubmission, RoundSpec
from tsflab.realtime.store import PanelStore
from tsflab.realtime.tracks import TrackSpec


def _history(spec: RoundSpec, store: PanelStore, steps: int) -> np.ndarray:
    cutoff = pd.Timestamp(spec.cutoff)
    cutoff = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
    offset = pd.tseries.frequencies.to_offset(spec.freq)
    index = pd.date_range(end=cutoff, periods=steps, freq=offset)
    frame = store.read(start=index[0], end=cutoff).reindex(index=index, columns=spec.channels)
    frame = frame.ffill().bfill()
    return frame.fillna(pd.Series(spec.norm_mean, index=spec.channels)).to_numpy(dtype=float)


def _lead(spec: RoundSpec) -> int:
    """Grid steps between the cutoff and the last target."""
    offset = pd.tseries.frequencies.to_offset(spec.freq)
    cutoff = pd.Timestamp(spec.cutoff)
    last = pd.Timestamp(spec.target_timestamps[-1])
    return len(pd.date_range(cutoff, last, freq=offset)) - 1


def naive(spec: RoundSpec, store: PanelStore, track: TrackSpec) -> np.ndarray:
    last = _history(spec, store, 1)[-1]
    return np.repeat(last[None, :], spec.horizon, axis=0)


def seasonal_naive(spec: RoundSpec, store: PanelStore, track: TrackSpec) -> np.ndarray:
    period = max(1, track.seasonal_period)
    history = _history(spec, store, period)
    total = _lead(spec)
    path = np.stack([history[(k % period)] for k in range(total)])
    return path[-spec.horizon :]


def window_mean(spec: RoundSpec, store: PanelStore, track: TrackSpec) -> np.ndarray:
    mean = _history(spec, store, spec.seq_len).mean(axis=0)
    return np.repeat(mean[None, :], spec.horizon, axis=0)


BASELINES = {"Naive": naive, "SeasonalNaive": seasonal_naive, "WindowMean": window_mean}


def run_baselines(
    spec: RoundSpec, store: PanelStore, track: TrackSpec, names: list[str] | None = None
) -> list[ForecastSubmission]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    out = []
    for name in names or list(BASELINES):
        values = BASELINES[name](spec, store, track)
        out.append(
            ForecastSubmission(
                track=spec.track,
                round_id=spec.round_id,
                model=name,
                submitter="tsflab-baselines",
                submitted_at=now,
                predictions=[[float(v) for v in row] for row in values],
            )
        )
    return out
