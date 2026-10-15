"""Real-time track specifications (``configs/realtime/<id>.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from tsflab.core.paths import repository_root

TRACK_DIR = Path("configs") / "realtime"


@dataclass(frozen=True)
class TrackSpec:
    id: str
    title: str
    mode: str
    freq: str
    seq_len: int
    horizon: int
    submission_hours: float
    seasonal_period: int
    min_coverage: float
    history_days: int | None = None
    tz: str = "UTC"
    source: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)


def _load(path: Path) -> TrackSpec:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    track = raw["track"]
    return TrackSpec(
        id=track["id"],
        title=track["title"],
        mode=track.get("mode", "time_series"),
        freq=track["freq"],
        seq_len=int(track["seq_len"]),
        horizon=int(track["horizon"]),
        submission_hours=float(track.get("submission_hours", 24)),
        seasonal_period=int(track.get("seasonal_period", 1)),
        min_coverage=float(track.get("min_coverage", 0.8)),
        history_days=track.get("history_days"),
        tz=track.get("tz", "UTC"),
        source=dict(raw.get("source", {})),
        bootstrap=dict(raw.get("bootstrap", {})),
    )


def list_tracks(root: Path | None = None) -> list[TrackSpec]:
    directory = (root or repository_root()) / TRACK_DIR
    return [_load(path) for path in sorted(directory.glob("*.toml"))]


def get_track(track_id: str, root: Path | None = None) -> TrackSpec:
    path = (root or repository_root()) / TRACK_DIR / f"{track_id}.toml"
    if not path.is_file():
        known = ", ".join(t.id for t in list_tracks(root)) or "none"
        raise KeyError(f"unknown real-time track {track_id!r} (known: {known})")
    spec = _load(path)
    if spec.id != track_id:
        raise ValueError(f"{path} declares id {spec.id!r}; the file name must match")
    return spec
