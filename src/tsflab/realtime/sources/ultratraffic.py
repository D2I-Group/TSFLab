"""Bootstrap a traffic track from the local UltraTraffic parquet store.

The store (``dataset/ultratraffic``, built by
:mod:`tsflab.data.prepare.ultratraffic`) is the history source of the four
``traffic_pems_*`` tracks; it has no static dataset presets of its own. The
bootstrap keeps the station set of the most recent year, so live PeMS
increments extend exactly the same channels.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from tsflab.data.ultratraffic_store import load_panel
from tsflab.realtime.tracks import TrackSpec


def load(track: TrackSpec) -> pd.DataFrame:
    root = Path(os.environ.get(track.bootstrap.get("root_env", "ULTRATRAFFIC_ROOT"), "dataset/ultratraffic"))
    region = track.bootstrap["region"]
    available = sorted(int(p.stem) for p in (root / region / "static").glob("[0-9][0-9][0-9][0-9].parquet"))
    if not available:
        raise FileNotFoundError(
            f"no UltraTraffic store for {region} under {root}; run "
            "`uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip`"
        )
    first = int(track.bootstrap.get("first_year", available[0]))
    years = [y for y in available if y >= first]
    return load_panel(str(root), region, years, "static", "last")
