"""GIFT-EVAL dataset: loads HuggingFace Arrow data with GIFT-EVAL split protocol."""

from __future__ import annotations

import math
import os
import re
from typing import Tuple

import datasets as hf_datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.schemas.datasets.gift_eval import GiftEvalParameterConfig

# ---------------------------------------------------------------------------
# Prediction-length maps (replicated from GIFT-EVAL to avoid runtime dep)
# ---------------------------------------------------------------------------

# Mirrors gift_eval's PRED_LENGTH_MAP (src/gift_eval/data.py). Upstream defines no
# quarterly/yearly entries (Q/A series come only from M4, see below), so none are
# added here; the GIFT-Eval paper publishes no values for them either.
_PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

_M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

_TEST_SPLIT = 0.1
_MAX_WINDOW = 20

# Current pandas aliases mapped back to the legacy short codes used by the
# GIFT-EVAL protocol maps above (and stored in the Arrow files).
_FREQ_ALIAS = {
    "Y": "A",
    "YE": "A",
    "YS": "A",
    "QE": "Q",
    "QS": "Q",
    "ME": "M",
    "MS": "M",
    "h": "H",
    "min": "T",
    "s": "S",
    "ms": "L",
    "us": "U",
    "ns": "N",
}

# Legacy fixed-width aliases stored in the GIFT-EVAL Arrow files -> pandas 3.
_LEGACY_FIXED = {"H": "h", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns"}

# Legacy and current period aliases -> (family, explicit anchor side or None).
_PERIOD_FAMILY = {
    "M": ("M", None), "ME": ("M", "E"), "MS": ("M", "S"),
    "Q": ("Q", None), "QE": ("Q", "E"), "QS": ("Q", "S"),
    "A": ("Y", None), "Y": ("Y", None), "YE": ("Y", "E"),
    "AS": ("Y", "S"), "YS": ("Y", "S"),
}

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

_FREQ_RE = re.compile(r"^(\d*)([A-Za-z]+)(?:-([A-Za-z]+))?$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_freq(freq_str: str) -> tuple[str, str, str | None]:
    match = _FREQ_RE.match(str(freq_str).strip())
    if match is None:
        raise ValueError(f"Unrecognised frequency: {freq_str!r}")
    mult, base, anchor = match.groups()
    return mult, base, anchor.upper() if anchor else None


def _norm_freq(freq_str: str) -> str:
    """Normalise a frequency string to the legacy GIFT-EVAL short code.

    Accepts legacy (``"5T"``, ``"Q-DEC"``, ``"A-DEC"``) and current
    (``"15min"``, ``"QE-DEC"``, ``"YE-DEC"``) spellings and returns the code used
    by the protocol maps (``"T"``, ``"Q"``, ``"A"``, ``"W"`` ...).
    """
    _, base, _ = _split_freq(freq_str)
    return _FREQ_ALIAS.get(base, base)


def _pandas_freq(freq_str: str, start: pd.Timestamp | None = None) -> str:
    """Translate a stored (possibly legacy) frequency to a pandas 3 alias.

    ``"H"`` -> ``"h"``, ``"5T"`` -> ``"5min"``, ``"10S"`` -> ``"10s"``,
    ``"Q-DEC"`` -> ``"QE-DEC"``, ``"A-DEC"`` -> ``"YE-DEC"``; ``"W-FRI"`` and
    ``"D"`` are unchanged. Legacy monthly/quarterly/yearly codes become the
    period-start alias (``MS``, ``QS-JAN``, ``YS-JAN``) when ``start`` lies on such
    a boundary, so the index begins exactly at the stored start; otherwise the
    period-end alias, which is what the legacy code meant.
    """
    mult, base, anchor = _split_freq(freq_str)
    if base in _LEGACY_FIXED:
        return f"{mult}{_LEGACY_FIXED[base]}" + (f"-{anchor}" if anchor else "")
    if base not in _PERIOD_FAMILY:
        return f"{mult}{base}" + (f"-{anchor}" if anchor else "")

    family, side = _PERIOD_FAMILY[base]
    if family == "M":
        if side is None:
            side = "S" if start is not None and _is_period_start(start, 1, 1) else "E"
        return f"{mult}M{side}"

    step = 3 if family == "Q" else 12
    if side == "S":
        return f"{mult}{family}S-{anchor or 'JAN'}"
    end_month = anchor or "DEC"
    if side is None and start is not None:
        first_month = (_MONTHS.index(end_month) + 1) % 12 + 1  # 1-based
        if _is_period_start(start, first_month, step):
            return f"{mult}{family}S-{_MONTHS[first_month - 1]}"
    return f"{mult}{family}E-{end_month}"


def _is_period_start(start: pd.Timestamp, first_month: int, step: int) -> bool:
    """True when ``start`` is midnight on the first day of a period."""
    if start != start.normalize() or start.day != 1:
        return False
    return (start.month - first_month) % step == 0


def _build_stamp(start: pd.Timestamp, freq: str, length: int) -> np.ndarray:
    """Generate (length, 6) float32 time-feature array.

    ``freq`` may be a legacy alias as stored in the GIFT-EVAL files; it is
    translated with :func:`_pandas_freq` before building the index.
    """
    dates = pd.date_range(
        start=start, periods=length, freq=_pandas_freq(freq, start)
    )
    return np.column_stack(
        [
            dates.year,
            dates.month,
            dates.day,
            dates.weekday,
            dates.hour,
            dates.minute,
        ]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class Dataset_GiftEval(Dataset):
    """Dataset backed by GIFT-EVAL HuggingFace Arrow files.

    Loads data via ``datasets.load_from_disk``, applies the GIFT-EVAL
    train/val/test splitting protocol.

    For **train** and **val** splits the series data and timestamps are
    stored contiguously and windows are computed on-the-fly in
    ``__getitem__`` (memory-efficient for large datasets like
    ``electricity/15T``).

    For **test** the small number of rolling evaluation windows is
    materialised directly.

    Parameters
    ----------
    root_path : str
        Root directory of GIFT-EVAL data (the ``GIFT_EVAL`` env-var path).
    data_path : str
        Sub-path under *root_path* identifying the dataset, e.g.
        ``"electricity/15T"`` or ``"m4_monthly"``.
    size : tuple[int, int, int]
        ``(seq_len, label_len, pred_len)``.
    flag : str
        ``"train"``, ``"val"``, or ``"test"``.
    features : str
        Feature mode (``"M"``, ``"S"``, ``"MS"``).  For univariate datasets
        ``"S"`` and ``"M"`` are equivalent.
    scale : bool
        Whether to apply StandardScaler (fitted on training data only).
    windows : int or None
        Number of rolling test windows.  ``None`` auto-calculates using
        the GIFT-EVAL formula.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "M",
        scale: bool = True,
        windows: int | None = None,
    ):
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.scale = scale

        # Load HuggingFace arrow dataset
        ds_path = os.path.join(os.path.expanduser(root_path), data_path)
        hf_ds = hf_datasets.load_from_disk(ds_path)

        # Parse series list
        self._is_m4 = "m4" in data_path.lower()
        freq = hf_ds[0]["freq"]
        self._freq = freq

        series_list, starts = self._parse_series(hf_ds, features)
        min_len = min(len(s) for s in series_list)

        # Compute windows
        if windows is not None:
            self._windows = windows
        elif self._is_m4:
            self._windows = 1
        else:
            self._windows = min(
                max(1, math.ceil(_TEST_SPLIT * min_len / self.pred_len)),
                _MAX_WINDOW,
            )

        # Fit scaler on training portions
        scaler = self._fit_scaler(series_list) if scale else None
        self.scaler: StandardScaler | None = scaler

        if flag in ("train", "val"):
            self._init_sliding(series_list, starts, flag, scaler)
        elif flag == "test":
            self._init_test(series_list, starts, scaler)
        else:
            raise ValueError(f"Invalid flag '{flag}'")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_sliding(
        self,
        series_list: list[np.ndarray],
        starts: list[pd.Timestamp],
        flag: str,
        scaler: StandardScaler | None,
    ) -> None:
        """Prepare contiguous per-series arrays for on-the-fly windowing."""
        self._series_data: list[np.ndarray] = []
        self._series_stamp: list[np.ndarray] = []
        self._cum_windows: list[int] = []  # cumulative window counts
        self._split_regions: list[np.ndarray] = []  # every series, windowed or not
        window_len = self.seq_len + self.pred_len
        total = 0

        for series, start in zip(series_list, starts):
            length = len(series)
            train_end, val_end = self._split_borders(length)

            if flag == "train":
                region = slice(0, train_end)
            else:  # val
                region = slice(max(0, train_end - self.seq_len), val_end)

            data = series[region]
            if scaler is not None:
                data = scaler.transform(data).astype(np.float32)
            else:
                data = data.astype(np.float32)
            self._split_regions.append(data)

            n_windows = len(data) - window_len + 1
            if n_windows <= 0:
                continue

            stamp = _build_stamp(start, self._freq, length)[region]
            self._series_data.append(data)
            self._series_stamp.append(stamp)
            total += n_windows
            self._cum_windows.append(total)

        self._total_len = total
        # test arrays not used in this mode
        self._test_x = self._test_y = self._test_xm = self._test_ym = None

    def _init_test(
        self,
        series_list: list[np.ndarray],
        starts: list[pd.Timestamp],
        scaler: StandardScaler | None,
    ) -> None:
        """Materialise rolling GIFT-EVAL test windows."""
        all_x, all_y, all_xm, all_ym = [], [], [], []
        self._split_regions = []

        for series, start in zip(series_list, starts):
            length = len(series)
            _, val_end = self._split_borders(length)

            data = scaler.transform(series).astype(np.float32) if scaler else series.astype(np.float32)
            stamp = _build_stamp(start, self._freq, length)
            self._split_regions.append(data[val_end:])

            for w in range(self._windows):
                forecast_start = val_end + w * self.pred_len
                forecast_end = forecast_start + self.pred_len
                context_end = forecast_start
                x_start = context_end - self.seq_len
                if x_start < 0:
                    continue
                y_start = context_end - self.label_len
                all_x.append(data[x_start:context_end])
                all_y.append(data[y_start:forecast_end])
                all_xm.append(stamp[x_start:context_end])
                all_ym.append(stamp[y_start:forecast_end])

        if not all_x:
            raise RuntimeError(
                "No test windows generated. "
                "Series may be too short for the requested seq_len + pred_len."
            )
        self._test_x = np.stack(all_x).astype(np.float32)
        self._test_y = np.stack(all_y).astype(np.float32)
        self._test_xm = np.stack(all_xm).astype(np.float32)
        self._test_ym = np.stack(all_ym).astype(np.float32)
        self._total_len = len(self._test_x)
        # sliding arrays not used in test mode
        self._series_data = []
        self._series_stamp = []
        self._cum_windows = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_series(
        hf_ds: hf_datasets.Dataset,
        features: str,
    ) -> tuple[list[np.ndarray], list[pd.Timestamp]]:
        """Convert HF rows to list of (length, C) arrays and start times."""
        series_list: list[np.ndarray] = []
        starts: list[pd.Timestamp] = []
        for row in hf_ds:
            target = row["target"]
            start = row["start"]
            if isinstance(start, (np.ndarray, list, tuple)):
                # Some files (e.g. restaurant) store the start as a 0-d or
                # one-element datetime array instead of a scalar.
                start = np.ravel(np.asarray(start, dtype=object))[0]
            start = pd.Timestamp(start)
            arr = np.asarray(target, dtype=np.float32)
            if arr.ndim == 2:
                # Multivariate: (dim, length) → (length, dim)
                arr = arr.T
                if features == "S":
                    arr = arr[:, :1]
            else:
                # Univariate: (length,) → (length, 1)
                arr = arr.reshape(-1, 1)
            series_list.append(arr)
            starts.append(start)
        return series_list, starts

    def _split_borders(self, length: int) -> tuple[int, int]:
        """Return (train_end, val_end) indices for a series of given length."""
        test_points = self.pred_len * self._windows
        val_points = self.pred_len  # one extra pred_len for validation
        val_end = length - test_points
        train_end = val_end - val_points
        return train_end, val_end

    def _fit_scaler(
        self,
        series_list: list[np.ndarray],
    ) -> StandardScaler:
        """Fit a StandardScaler on the training portions of all series."""
        train_chunks = []
        for series in series_list:
            train_end, _ = self._split_borders(len(series))
            if train_end > 0:
                train_chunks.append(series[:train_end])
        scaler = StandardScaler()
        scaler.fit(np.concatenate(train_chunks, axis=0))
        return scaler

    def _resolve_index(self, index: int) -> tuple[int, int]:
        """Map a global index to (series_idx, local_window_offset)."""
        import bisect

        series_idx = bisect.bisect_right(self._cum_windows, index)
        prev = self._cum_windows[series_idx - 1] if series_idx > 0 else 0
        local = index - prev
        return series_idx, local

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int) -> Tuple:
        if self.flag == "test":
            return (
                self._test_x[index],
                self._test_y[index],
                self._test_xm[index],
                self._test_ym[index],
            )

        # On-the-fly sliding window for train / val
        si, offset = self._resolve_index(index)
        data = self._series_data[si]
        stamp = self._series_stamp[si]

        x_end = offset + self.seq_len
        y_start = x_end - self.label_len
        y_end = x_end + self.pred_len

        return (
            data[offset:x_end],
            data[y_start:y_end],
            stamp[offset:x_end],
            stamp[y_start:y_end],
        )

    def series_array(self) -> np.ndarray:
        """Return this split's observed values as one ``(T, C)`` array.

        Analysis accessor (``tsf data analyze``), independent of the window
        count: series too short for ``seq_len + pred_len`` are still included.
        Each GIFT-EVAL series contributes its own channel(s). Series of unequal
        length are right-aligned (most recent rows) to a common length ``L``
        chosen to cover the most values (``L`` x number of series at least ``L``
        long); shorter series are left out.
        """
        regions = [r for r in self._split_regions if len(r)]
        if not regions:
            return np.zeros((0, 1), dtype=np.float32)
        lengths = sorted((len(r) for r in regions), reverse=True)
        # With lengths sorted descending, (i + 1) series are at least lengths[i] long.
        best = max(range(len(lengths)), key=lambda i: lengths[i] * (i + 1))
        common = lengths[best]
        kept = [r[len(r) - common:] for r in regions if len(r) >= common]
        return np.concatenate(kept, axis=1)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the GIFT-EVAL dataset under the name ``"gift_eval"``."""
    DATASET_REGISTRY.register(
        "gift_eval", Dataset_GiftEval, GiftEvalParameterConfig,
        task_modes=frozenset({"time_series"}),
        storage="selector",
    )
