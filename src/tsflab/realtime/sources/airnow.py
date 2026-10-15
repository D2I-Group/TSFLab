"""EPA AirNow hourly observations from the public file products (no key).

``files.airnowtech.org/airnow/<year>/<yyyymmdd>/HourlyData_<yyyymmddhh>.dat`` holds
one UTC hour of preliminary, unvalidated observations from all reporting
agencies (pipe-delimited, no header). The files are served without credentials
and the archive reaches back years, so a track needs no AirNow API key.
Channels are US monitors (AQSID prefix ``840``) reporting one parameter,
default PM2.5 in ug/m3; the bootstrap keeps the best-covered ``max_sites``.
AirNow data are preliminary and may be revised later by AQS; the panel keeps
the first value it saw, like every real-time track.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tsflab.realtime.tracks import TrackSpec

BASE = "https://files.airnowtech.org/airnow"
COLUMNS = ["date", "time", "aqsid", "site", "gmt_offset", "parameter", "unit", "value", "agency"]


def hour_url(hour: pd.Timestamp) -> str:
    return f"{BASE}/{hour:%Y}/{hour:%Y%m%d}/HourlyData_{hour:%Y%m%d%H}.dat"


def parse_hourly_file(text: str, parameter: str = "PM2.5", prefix: str = "840") -> pd.Series:
    """One ``HourlyData`` file -> values indexed by (UTC timestamp, AQSID) for one parameter."""
    frame = pd.read_csv(
        io.StringIO(text),
        sep="|",
        header=None,
        names=COLUMNS,
        dtype={"aqsid": str},
        usecols=range(len(COLUMNS)),
        on_bad_lines="skip",
    )
    frame = frame[(frame["parameter"] == parameter) & frame["aqsid"].str.startswith(prefix, na=False)]
    stamp = pd.to_datetime(frame["date"] + " " + frame["time"], format="%m/%d/%y %H:%M", errors="coerce")
    value = pd.to_numeric(frame["value"], errors="coerce")
    keep = stamp.notna() & value.notna()
    series = pd.Series(value[keep].to_numpy(), index=pd.MultiIndex.from_arrays([stamp[keep], frame["aqsid"][keep]]))
    return series[~series.index.duplicated(keep="last")]


def _download(hour: pd.Timestamp) -> str | None:
    import requests

    for attempt in range(3):
        try:
            response = requests.get(hour_url(hour), timeout=60)
        except requests.RequestException:
            continue
        if response.status_code in (403, 404):  # hour not published (yet)
            return None
        if response.ok:
            return response.content.decode("utf-8", errors="replace")
    return None


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    parameter = track.source.get("parameter", "PM2.5")
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    with ThreadPoolExecutor(max_workers=int(track.source.get("workers", 8))) as pool:
        texts = list(pool.map(_download, hours))
    parts = [parse_hourly_file(text, parameter, track.source.get("aqsid_prefix", "840")) for text in texts if text]
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts).unstack("aqsid").sort_index()
    panel.index.name = panel.columns.name = None
    if channels:
        return panel.reindex(columns=channels)
    coverage = panel.notna().mean().sort_values(ascending=False, kind="stable")
    keep = coverage[coverage >= float(track.source.get("min_coverage", 0.9))]
    return panel[sorted(keep.index[: int(track.source.get("max_sites", 200))])]
