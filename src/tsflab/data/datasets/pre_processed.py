"""Pre-processed dataset: loads pre-windowed train/val/test from .npz files."""

from __future__ import annotations

import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.schemas.datasets.pre_processed import PreProcessedParameterConfig

_FLAG_TO_FILE = {"train": "train.npz", "val": "val.npz", "test": "test.npz"}


class Dataset_PreProcessed(Dataset):
    """Dataset backed by pre-windowed ``.npz`` split files.

    Each ``.npz`` file must contain:

    - ``x``       — float32 ``(N, seq_len, n_features)`` input series
    - ``y``       — float32 ``(N, label_len + pred_len, n_features)`` decoder target
    - ``x_mark``  — float32 ``(N, seq_len, 6)`` input timestamps
    - ``y_mark``  — float32 ``(N, label_len + pred_len, 6)`` output timestamps
    - ``scaler_mean``  — float32 ``(n_features,)`` *(optional, saved by presplit_to_npy)*
    - ``scaler_scale`` — float32 ``(n_features,)`` *(optional)*

    Generate the files with ``tsf data prepare``.

    Parameters
    ----------
    root_path : str
        Directory containing ``train.npz``, ``val.npz``, and ``test.npz``.
    data_path : str
        Unused internal loader slot; the public preset points at the directory.
    size : tuple[int, int, int]
        Common loader window contract; files already contain these windows.
    flag : str, optional
        Split to load: ``"train"``, ``"val"``, or ``"test"``.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "M",
    ):
        super().__init__()
        del data_path, features
        self.seq_len, self.label_len, self.pred_len = size
        if flag not in _FLAG_TO_FILE:
            raise ValueError(f"Invalid flag '{flag}'. Expected one of {list(_FLAG_TO_FILE)}.")

        split_file = os.path.join(root_path, _FLAG_TO_FILE[flag])
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Expected file not found: {split_file}. Generate npz splits with `tsf data prepare`."
            )

        data = np.load(split_file)
        self.x: np.ndarray = data["x"]
        self.y: np.ndarray = data["y"]
        self.x_mark: np.ndarray = data["x_mark"]
        self.y_mark: np.ndarray = data["y_mark"]

        if "scaler_mean" in data:
            self.scaler: StandardScaler | None = StandardScaler()
            self.scaler.mean_ = data["scaler_mean"]
            self.scaler.scale_ = data["scaler_scale"]
            self.scaler.var_ = data["scaler_scale"] ** 2
            self.scaler.n_features_in_ = len(data["scaler_mean"])
            self.scaler.n_samples_seen_ = 0
        else:
            self.scaler = None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple:
        return self.x[index], self.y[index], self.x_mark[index], self.y_mark[index]

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)


def register() -> None:
    """Register the pre-processed dataset under the name ``"pre_processed"``."""
    DATASET_REGISTRY.register(
        "pre_processed",
        Dataset_PreProcessed,
        PreProcessedParameterConfig,
        task_modes=frozenset({"time_series"}),
        storage="directory",
    )
