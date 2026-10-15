"""Data sources for real-time tracks.

Every source returns a wide ``pandas.DataFrame`` whose index is a naive
timestamp in the track's local time convention and whose columns are channel
ids. Credentials are read from the environment variables named in the track's
``[source].credentials`` list and never written to disk.
"""

from __future__ import annotations

import os

import pandas as pd

from tsflab.realtime.tracks import TrackSpec


def require_env(names: list[str]) -> dict[str, str]:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing credentials: set " + ", ".join(missing) + " in the environment")
    return {name: os.environ[name] for name in names}


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    """Fetch observations in ``[start, end]`` for ``track`` from its live source."""
    kind = track.source.get("kind")
    if kind == "akshare_csi300":
        from .akshare_csi300 import fetch as run
    elif kind == "nasdaq100":
        from .nasdaq100 import fetch as run
    elif kind == "pems":
        from .pems import fetch as run
    elif kind == "sp500":
        from .sp500 import fetch as run
    elif kind == "openmeteo":
        from .openmeteo import fetch as run
    elif kind == "airnow":
        from .airnow import fetch as run
    elif kind == "ercot":
        from .ercot import fetch as run
    else:
        raise ValueError(f"track {track.id!r}: unknown source kind {kind!r}")
    return run(track, start, end, channels)


def bootstrap(track: TrackSpec) -> pd.DataFrame:
    """Build the initial history for ``track`` from its archive."""
    kind = track.bootstrap.get("kind")
    if kind == "ultratraffic":
        from .ultratraffic import load as run
    elif kind == "hub_panel":
        from .hub_panel import load as run
    elif kind == "source":
        end = pd.Timestamp.now(tz=track.tz).tz_localize(None).floor("h")
        if "start" in track.bootstrap:
            start = pd.Timestamp(track.bootstrap["start"])
        else:
            start = end - pd.Timedelta(days=int(track.bootstrap.get("days", 365)))
        return fetch(track, start, end, None)
    else:
        raise ValueError(f"track {track.id!r}: unknown bootstrap kind {kind!r}")
    return run(track)
