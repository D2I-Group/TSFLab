"""Calendar covariates shared by loaders and real-time exports.

Both are known in advance for any future timestamp, so they may be used as
future covariates without leaking target information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Fraction of the day elapsed, in ``[0, 1)``."""
    return ((index.hour * 60 + index.minute) / (24 * 60)).to_numpy(np.float32)


def day_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Weekday scaled to ``[0, 1)`` (Monday = 0)."""
    return (index.dayofweek / 7.0).to_numpy(np.float32)


def node_calendar(index: pd.DatetimeIndex, num_nodes: int) -> np.ndarray:
    """``(T, N, 2)`` time-of-day and day-of-week covariates broadcast to every node."""
    features = np.stack([time_of_day(index), day_of_week(index)], axis=-1)  # (T, 2)
    return np.repeat(features[:, None, :], num_nodes, axis=1)


def time_marks(index: pd.DatetimeIndex) -> np.ndarray:
    """``(T, 6)`` year, month, day, weekday, hour, minute marks.

    Same layout as the CSV loaders' ``_build_time_stamp`` (raw integers).
    """
    return np.column_stack([index.year, index.month, index.day, index.dayofweek, index.hour, index.minute]).astype(
        np.float32
    )
