"""Shared fetch loop for daily equity panels (one channel per symbol).

A source supplies its constituent list and a per-symbol adjusted-close fetcher;
this module adds the per-request resume cache, pacing, and the log-return
transform so every stock track stores the same kind of values.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tsflab.realtime.tracks import TrackSpec

CloseFetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]


def with_fallback(providers: list[Callable[..., pd.Series]], *args, attempts: int = 4, label: str = "") -> pd.Series:
    """Call providers in order, retrying with backoff; move the one that works first.

    ``providers`` is reordered in place, so a blocked vendor costs one failed
    request per run rather than one per symbol.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        for provider in list(providers):
            try:
                series = provider(*args)
            except Exception as exc:  # vendors raise heterogeneous network/parse errors
                last_error = exc
                continue
            if providers[0] is not provider:
                providers.remove(provider)
                providers.insert(0, provider)
            return series
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {label}: {last_error}")


def _cache_dir(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """Per-request cache so an interrupted fetch resumes instead of restarting."""
    root = Path(os.environ.get("TSFLAB_REALTIME_ROOT", "dataset/realtime"))
    return root / "_cache" / track.id / f"{start:%Y%m%d}-{end:%Y%m%d}"


def fetch_panel(
    track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str], daily_close: CloseFetcher
) -> pd.DataFrame:
    cache = _cache_dir(track, start, end)
    cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for number, symbol in enumerate(symbols, start=1):
        path = cache / f"{symbol}.parquet"
        if path.is_file():  # already fetched by an earlier, interrupted run
            series[symbol] = pd.read_parquet(path)["value"]
            continue
        if number % 25 == 0:
            print(f"  fetched {number}/{len(symbols)} symbols", flush=True)
        # one extra week so the first requested day has a previous close
        close = daily_close(symbol, start - pd.Timedelta(days=7), end)
        if len(close):
            close = close[~close.index.duplicated(keep="last")].sort_index()
            if track.source.get("transform", "log_return") == "log_return":
                close = np.log(close).diff()
            series[symbol] = close.loc[start:end]
            series[symbol].to_frame("value").to_parquet(path)
        time.sleep(float(track.source.get("pause_seconds", 0.5)))
    return pd.DataFrame(series).sort_index()
