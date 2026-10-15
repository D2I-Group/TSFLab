"""Read the local UltraTraffic parquet store (torch-free).

Read by the real-time traffic tracks (``traffic_pems_*``) to bootstrap their history.
"""

from __future__ import annotations

import os

import pandas as pd

_FILES = {
    "static": "static/{year}.parquet",
    "cl_common": "cl/{year}_common.parquet",
    "cl_added": "cl/{year}_added.parquet",
}


def load_panel(root: str, region: str, years: list[int], variant: str, stations: str) -> pd.DataFrame:
    """Concatenate yearly panels into one ``(time, station)`` frame."""
    frames = []
    for year in years:
        path = os.path.join(root, region, _FILES[variant].format(year=year))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} is missing; build the store with "
                "`uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip`"
            )
        frames.append(pd.read_parquet(path))
    if stations == "intersection":
        columns = sorted(set.intersection(*(set(f.columns) for f in frames)))
    else:
        columns = list(frames[-1].columns)
    frames = [f.reindex(columns=columns).astype("float32") for f in frames]
    panel = pd.concat([f for f in frames if not f.empty]).sort_index()
    return panel[~panel.index.duplicated()]
