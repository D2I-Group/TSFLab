"""Open-Meteo hourly weather at fixed sites (free, no key, CC BY 4.0).

One variable per track (``source.variable``, e.g. ``temperature_2m`` or
``shortwave_radiation``) at the sites listed in ``source.sites`` as
``[id, latitude, longitude]``. Values come from the Historical Forecast API,
which stitches the analysis hours of the operational weather models and is the
same product the live forecast endpoint serves for the most recent hours, so
bootstraps and weekly updates share one definition. (The ERA5 archive endpoint
lags by days and is a different product.) Timestamps are UTC. The newest hours
of the response are model forecasts rather than observations, so values are
clipped to ``lag_hours`` before now.
"""

from __future__ import annotations

import time

import pandas as pd

from tsflab.realtime.tracks import TrackSpec

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def parse_hourly(payload, variable: str, ids: list[str]) -> pd.DataFrame:
    """Open-Meteo multi-location JSON (list, or dict for one site) -> wide frame by site id."""
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(ids):
        raise ValueError(f"expected {len(ids)} locations, got {len(locations)}")
    series = {}
    for site, location in zip(ids, locations):
        hourly = location["hourly"]
        series[site] = pd.Series(
            [float(v) if v is not None else float("nan") for v in hourly[variable]],
            index=pd.to_datetime(hourly["time"]),
            dtype="float64",
        )
    frame = pd.DataFrame(series).sort_index()
    frame.index.name = None
    return frame


def _get(params: dict) -> object:
    import requests

    for attempt in range(5):
        response = requests.get(API, params=params, timeout=90)
        if response.status_code == 429:  # per-minute or hourly quota: back off
            time.sleep(30 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Open-Meteo rate limit persisted")


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    variable = track.source["variable"]
    sites = {site[0]: (site[1], site[2]) for site in track.source["sites"]}
    if channels:
        sites = {k: v for k, v in sites.items() if k in set(channels)}
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    end = min(end, (now - pd.Timedelta(hours=int(track.source.get("lag_hours", 3)))).floor("h"))
    if end < start:
        return pd.DataFrame()
    batch, window = int(track.source.get("batch", 10)), int(track.source.get("window_days", 120))
    ids = list(sites)
    frames = []
    for lo in range(0, len(ids), batch):
        chunk = ids[lo : lo + batch]
        parts = []
        for day in pd.date_range(start.normalize(), end.normalize(), freq=f"{window}D"):
            last = min(day + pd.Timedelta(days=window - 1), end.normalize())
            payload = _get(
                {
                    "latitude": ",".join(str(sites[s][0]) for s in chunk),
                    "longitude": ",".join(str(sites[s][1]) for s in chunk),
                    "hourly": variable,
                    "start_date": f"{day:%Y-%m-%d}",
                    "end_date": f"{last:%Y-%m-%d}",
                    "timezone": "GMT",
                    "cell_selection": "nearest",
                }
            )
            parts.append(parse_hourly(payload, variable, chunk))
            time.sleep(float(track.source.get("pause_seconds", 1.0)))
        frames.append(pd.concat(parts))
    panel = pd.concat(frames, axis=1).sort_index()
    return panel.loc[start:end]
