"""Real-time tracks served as static datasets.

A track's append-only :class:`~tsflab.realtime.store.PanelStore` is read at a
fixed release (``version``) or pinned Hub ``revision`` and exposed under the
standard dataset contract, so every catalog model can be trained and scored on
the same frozen snapshot that a real-time round used as history. Two names
share this class, mirroring the CauAir pair:

* ``realtime_panel_ts`` - channels are the panel columns; items carry the
  six calendar marks of the base loader;
* ``realtime_panel_st`` - spatiotemporal layout of ``forecast.export_bundle``:
  value ``(T, N)`` plus time-of-day / day-of-week covariates ``(T, N, 2)``.

Gaps are filled causally (forward fill, then back fill only at a series'
start), the split is chronological (default 7:1:2) and the z-score uses one
mean/std from the training rows, as in ``export_bundle``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.calendar import node_calendar
from tsflab.data.schemas.datasets.realtime_panel import DatasetParameterConfig


def _open_store(root_path: str, track: str, revision: str | None, repo_id: str | None):
    from tsflab.realtime.store import PanelStore

    base = Path(root_path).expanduser()
    if base.name != track:
        raise ValueError(f"dataset.path must end with the track id {track!r}; got {str(base)!r}")
    if revision is None:
        store = PanelStore(track, root=base.parent)
        if not store.exists:
            raise FileNotFoundError(
                f"no local store for track {track!r} at {base}; bootstrap or update it with "
                f"`tsf realtime`, or set dataset.params.revision to pull a Hub revision"
            )
        return store
    from tsflab.realtime.publish import DEFAULT_DATASET_REPO, pull_track

    store = PanelStore(track, root=base.parent / "_hub" / revision[:12])
    if not store.exists and not pull_track(store, repo_id or DEFAULT_DATASET_REPO, revision):
        raise FileNotFoundError(f"track {track!r} not found at Hub revision {revision!r}")
    return store


def load_panel(
    root_path: str,
    track: str,
    version: str | None = None,
    revision: str | None = None,
    repo_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """Return the frozen, gap-filled ``(T, N)`` panel on a regular grid."""
    from tsflab.realtime.tracks import get_track

    store = _open_store(root_path, track, revision, repo_id)
    manifest = store.manifest()
    cut = None
    if version is not None:
        releases = {r["version"]: r for r in manifest["releases"]}
        if version not in releases:
            raise ValueError(f"track {track!r} has no release {version!r}; known: {sorted(releases)}")
        release = releases[version]
        cut = pd.Timestamp(release["last_timestamp"])
        if release is manifest["releases"][-1] and store.content_hash() != release["content_sha256"]:
            raise ValueError(f"store content of {track!r} does not match release {version!r}")
    stop = pd.Timestamp(end) if end else None
    if cut is not None:
        stop = cut if stop is None else min(stop, cut)
    panel = store.read(start=start, end=stop)
    if panel.empty:
        raise ValueError(f"track {track!r} holds no rows in the requested range")
    grid = pd.date_range(panel.index.min(), panel.index.max(), freq=freq or get_track(track).freq)
    panel = panel.reindex(grid).ffill().bfill().fillna(0.0)
    return panel


class _RealtimePanelBase(Dataset):
    spatiotemporal = True

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "M",
        track: str = "",
        version: str | None = None,
        revision: str | None = None,
        repo_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
        scale: bool = True,
        calendar: bool = True,
        max_windows: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size
        panel = load_panel(root_path, track, version, revision, repo_id, start, end)
        values = panel.to_numpy(np.float32)
        total = len(values)
        cuts = np.cumsum(split_ratio) / sum(split_ratio)
        train_end, val_end = int(cuts[0] * total), int(cuts[1] * total)
        if scale:
            train = values[:train_end]
            self.value_mean = float(train.mean())
            self.value_std = float(train.std()) or 1.0
            values = (values - self.value_mean) / self.value_std
        else:
            self.value_mean = self.value_std = None
        self.values = values
        self.index = panel.index
        self.channels = list(panel.columns)
        self.num_nodes = values.shape[1]
        self.adj_mx = None
        self.marks = self._marks(panel.index, calendar)
        start_row, end_row = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, total)}[flag]
        # A window centred at ``c`` predicts rows ``c+1 .. c+pred_len``: the first
        # target of val/test is the first split row, its input borrows earlier rows.
        first = self.seq_len - 1 if flag == "train" else max(start_row - 1, self.seq_len - 1)
        centers = np.arange(first, end_row - self.pred_len)
        if max_windows is not None and len(centers) > max_windows:
            centers = centers[np.linspace(0, len(centers) - 1, max_windows).astype(np.int64)]
        self.idx = centers
        self._split_rows = (start_row, end_row)

    def series_array(self) -> np.ndarray:
        """Return this split's (scaled) panel rows as a ``(T, N)`` array."""
        start_row, end_row = self._split_rows
        return self.values[start_row:end_row]

    def _marks(self, index: pd.DatetimeIndex, calendar: bool) -> np.ndarray | None:  # pragma: no cover
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.idx)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.value_mean is None:
            return data
        return data * self.value_std + self.value_mean


class Dataset_RealtimePanel_ST(_RealtimePanelBase):
    """Spatiotemporal layout: ``(value_hist, value_fut, cov_hist, cov_fut)``."""

    spatiotemporal = True

    def _marks(self, index, calendar):
        if not calendar:
            return np.zeros((len(index), self.num_nodes, 0), np.float32)
        return node_calendar(index, self.num_nodes)

    def __getitem__(self, i: int) -> tuple:
        c = int(self.idx[i])
        h0 = c - self.seq_len + 1
        return (
            np.ascontiguousarray(self.values[h0 : c + 1]),
            np.ascontiguousarray(self.values[c + 1 : c + 1 + self.pred_len]),
            np.ascontiguousarray(self.marks[h0 : c + 1]),
            np.ascontiguousarray(self.marks[c + 1 : c + 1 + self.pred_len]),
        )


class Dataset_RealtimePanel_TS(_RealtimePanelBase):
    """Plain time-series layout: channels plus the six base calendar marks."""

    spatiotemporal = False

    def _marks(self, index, calendar):
        stamp = pd.DataFrame(
            {
                "year": index.year,
                "month": index.month,
                "day": index.day,
                "weekday": index.weekday,
                "hour": index.hour,
                "minute": index.minute,
            }
        )
        marks = stamp.to_numpy(np.float32)
        return marks if calendar else np.zeros_like(marks)

    def __getitem__(self, i: int) -> tuple:
        c = int(self.idx[i])
        h0 = c - self.seq_len + 1
        return (
            np.ascontiguousarray(self.values[h0 : c + 1]),
            np.ascontiguousarray(self.values[c + 1 : c + 1 + self.pred_len]),
            np.ascontiguousarray(self.marks[h0 : c + 1]),
            np.ascontiguousarray(self.marks[c + 1 : c + 1 + self.pred_len]),
        )


def register() -> None:
    """Register the real-time panel datasets."""
    DATASET_REGISTRY.register(
        "realtime_panel_st",
        Dataset_RealtimePanel_ST,
        DatasetParameterConfig,
        task_modes=frozenset({"spatiotemporal", "covariate"}),
        storage="directory",
    )
    DATASET_REGISTRY.register(
        "realtime_panel_ts",
        Dataset_RealtimePanel_TS,
        DatasetParameterConfig,
        task_modes=frozenset({"time_series"}),
        storage="directory",
    )
