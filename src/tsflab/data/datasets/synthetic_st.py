"""Synthetic spatiotemporal dataset for smoke-testing the spatiotemporal mode.

Generates a small ``(T, N, 3)`` tensor whose channels are
``[value, time_in_day, day_in_week]`` — the calendar-covariate convention the
spatiotemporal models (BiST, MAGE, STOP) consume. The value channel mixes a
daily and weekly sinusoid per node so the calendar covariates are informative.

Returns the spatiotemporal item contract: ``(value_hist, value_fut,
cov_hist, cov_fut)`` with value ``(T, N)`` and covariates ``(T, N, F)``.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.schemas.datasets.synthetic_st import DatasetParameterConfig


class Dataset_Synthetic_ST(Dataset):
    """Synthetic node-structured dataset with calendar covariates."""

    spatiotemporal = True

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "M",
        num_nodes: int = 8,
        steps_per_day: int = 24,
        length: int = 600,
        split_ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size
        self.num_nodes = num_nodes
        self.steps_per_day = steps_per_day
        self._build(flag, num_nodes, steps_per_day, length, split_ratio)

    def _build(self, flag, n, spd, length, split_ratio):
        rng = np.random.default_rng(0)
        t = np.arange(length)
        tod = (t % spd) / spd  # time-in-day in [0, 1)
        dow = ((t // spd) % 7) / 7.0  # day-in-week in [0, 1)
        # Per-node value: daily + weekly seasonality + noise.
        phase = np.linspace(0, np.pi, n)[None, :]
        value = (
            np.sin(2 * np.pi * tod[:, None] + phase)
            + 0.5 * np.sin(2 * np.pi * (t // spd)[:, None] / 7.0)
            + 0.1 * rng.standard_normal((length, n))
        ).astype(np.float32)
        cov = np.stack(
            [np.repeat(tod[:, None], n, axis=1), np.repeat(dow[:, None], n, axis=1)],
            axis=-1,
        ).astype(np.float32)  # (T, N, 2)
        data = np.concatenate([value[..., None], cov], axis=-1)  # (T, N, 3)

        borders = {
            "train": (0, int(split_ratio[0] * length)),
            "val": (
                int(split_ratio[0] * length) - self.seq_len,
                int((split_ratio[0] + split_ratio[1]) * length),
            ),
            "test": (
                int((split_ratio[0] + split_ratio[1]) * length) - self.seq_len,
                length,
            ),
        }
        b1, b2 = borders[flag]
        self.data = data[max(b1, 0) : b2]

    def __len__(self) -> int:
        """Number of windows in this split."""
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple:
        """Return ``(value_hist, value_fut, cov_hist, cov_fut)``."""
        s = index
        e = s + self.seq_len
        o = e + self.pred_len
        hist, fut = self.data[s:e], self.data[e:o]
        return (
            np.ascontiguousarray(hist[..., 0]),
            np.ascontiguousarray(fut[..., 0]),
            np.ascontiguousarray(hist[..., 1:]),
            np.ascontiguousarray(fut[..., 1:]),
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """No scaling is applied; returns data unchanged."""
        return data


def register() -> None:
    """Register the synthetic spatiotemporal dataset."""
    DATASET_REGISTRY.register(
        "synthetic_st",
        Dataset_Synthetic_ST,
        DatasetParameterConfig,
        task_modes=frozenset({"spatiotemporal", "covariate"}),
        storage="directory",
    )
