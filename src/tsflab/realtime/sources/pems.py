"""Caltrans PeMS clearinghouse (free account; ``PEMS_USER`` / ``PEMS_PASSWORD``).

Daily ``station_5min`` files are published the day after they are recorded,
so the weekly release aggregates them to hourly totals. File rows have no
header; the columns used here are the timestamp (0), station id (1), and total
flow (9), matching the station_hour flow values of the UltraTraffic archive.
"""

from __future__ import annotations

import gzip
import io

import pandas as pd

from tsflab.realtime.sources import require_env
from tsflab.realtime.tracks import TrackSpec

BASE = "https://pems.dot.ca.gov/"
TIMESTAMP, STATION, TOTAL_FLOW = 0, 1, 9


def parse_station_5min(payload: bytes) -> pd.DataFrame:
    """Parse one (optionally gzipped) station_5min file into hourly flow per station."""
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    frame = pd.read_csv(
        io.BytesIO(payload),
        header=None,
        usecols=[TIMESTAMP, STATION, TOTAL_FLOW],
        names=None,
        dtype={STATION: str},
    )
    frame.columns = ["timestamp", "station", "flow"]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%m/%d/%Y %H:%M:%S")
    frame["hour"] = frame["timestamp"].dt.floor("h")
    hourly = frame.groupby(["hour", "station"])["flow"].sum(min_count=1).unstack("station")
    hourly.index.name = None
    return hourly


def _session():
    import requests

    creds = require_env(["PEMS_USER", "PEMS_PASSWORD"])
    session = requests.Session()
    response = session.post(
        BASE,
        data={"username": creds["PEMS_USER"], "password": creds["PEMS_PASSWORD"], "login": "Login"},
        timeout=60,
    )
    response.raise_for_status()
    if "logout" not in response.text.lower():
        raise RuntimeError("PeMS login failed; check PEMS_USER / PEMS_PASSWORD")
    return session


def _daily_files(session, district: int, year: int) -> list[dict]:
    response = session.get(
        BASE,
        params={
            "srq": "clearinghouse",
            "district_id": district,
            "geotag": "null",
            "yy": year,
            "type": "station_5min",
            "returnformat": "text",
        },
        timeout=60,
    )
    response.raise_for_status()
    listing = response.json().get("data", {})
    return [entry for month in listing.values() for entry in month]


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    district = int(track.source["district"])
    session = _session()
    days = {d.strftime("%Y_%m_%d") for d in pd.date_range(start.normalize(), end.normalize(), freq="D")}
    frames = []
    for year in sorted({int(d[:4]) for d in days}):
        for entry in _daily_files(session, district, year):
            name = entry.get("file_name", "")
            if not any(day in name for day in days):
                continue
            response = session.get(BASE.rstrip("/") + entry["url"], timeout=300)
            response.raise_for_status()
            frames.append(parse_station_5min(response.content))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames).groupby(level=0).sum(min_count=1).sort_index()
    panel = panel.loc[start:end]
    return panel.reindex(columns=channels) if channels else panel
