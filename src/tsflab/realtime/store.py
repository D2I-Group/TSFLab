"""Versioned, append-only panel store for one real-time track.

Layout (local directory, mirrored to a Hugging Face dataset)::

    <root>/<track>/
        panel/<year>.parquet    # index = timestamp, columns = channel ids
        manifest.json           # channels, last timestamp, release history

A release never rewrites observed history: ``append`` only adds timestamps
after the stored maximum or fills cells that were missing, so every earlier
round remains reproducible from the release it was opened on.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def default_store_root() -> Path:
    override = os.environ.get("TSFLAB_REALTIME_ROOT")
    return Path(override).expanduser() if override else Path("dataset") / "realtime"


class PanelStore:
    def __init__(self, track: str, root: Path | None = None) -> None:
        self.track = track
        self.directory = (root or default_store_root()) / track
        self.panel_dir = self.directory / "panel"
        self.manifest_path = self.directory / "manifest.json"

    # -- manifest -----------------------------------------------------------
    def manifest(self) -> dict:
        if not self.manifest_path.is_file():
            return {"track": self.track, "channels": [], "last_timestamp": None, "releases": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.manifest_path)

    @property
    def exists(self) -> bool:
        return self.manifest_path.is_file()

    # -- data ---------------------------------------------------------------
    def read(self, start=None, end=None) -> pd.DataFrame:
        """Return the stored panel between ``start`` and ``end`` (inclusive)."""
        frames = []
        for path in sorted(self.panel_dir.glob("*.parquet")):
            year = int(path.stem)
            if start is not None and year < pd.Timestamp(start).year:
                continue
            if end is not None and year > pd.Timestamp(end).year:
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=self.manifest()["channels"], dtype="float64")
        panel = pd.concat(frames).sort_index()
        panel.index = pd.DatetimeIndex(panel.index)
        return panel.loc[start:end] if (start is not None or end is not None) else panel

    def append(self, new: pd.DataFrame, *, note: str = "") -> dict:
        """Merge ``new`` into the panel and record a release. Returns the release entry."""
        if new.empty:
            raise ValueError("refusing to record an empty release")
        new = new.copy()
        new.index = (
            pd.DatetimeIndex(new.index).tz_localize(None) if new.index.tz is not None else pd.DatetimeIndex(new.index)
        )
        new.columns = [str(c) for c in new.columns]
        manifest = self.manifest()
        channels = manifest["channels"] or sorted(new.columns)
        unknown = sorted(set(new.columns) - set(channels))
        if unknown:
            # New sensors are ignored rather than silently widening the panel:
            # changing the channel set would break comparability across rounds.
            new = new.drop(columns=unknown)
        new = new.reindex(columns=channels).astype("float64")
        self.panel_dir.mkdir(parents=True, exist_ok=True)
        for year, chunk in new.groupby(new.index.year):
            path = self.panel_dir / f"{year}.parquet"
            if path.is_file():
                old = pd.read_parquet(path)
                old.index = pd.DatetimeIndex(old.index)
                merged = old.combine_first(chunk)  # observed cells are never overwritten
            else:
                merged = chunk
            merged.sort_index().to_parquet(path)
        last = (
            max(pd.Timestamp(manifest["last_timestamp"]), new.index.max())
            if manifest["last_timestamp"]
            else new.index.max()
        )
        release = {
            "version": datetime.now(UTC).strftime("%Y.%m.%d-%H%M"),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows_added": int(len(new)),
            "last_timestamp": last.isoformat(),
            "content_sha256": self.content_hash(),
            "note": note,
        }
        manifest.update(channels=channels, last_timestamp=last.isoformat())
        manifest["releases"].append(release)
        self._write_manifest(manifest)
        return release

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.panel_dir.glob("*.parquet")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def latest_release(self) -> dict | None:
        releases = self.manifest()["releases"]
        return releases[-1] if releases else None
