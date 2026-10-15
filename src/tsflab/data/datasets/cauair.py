"""CauAir-style spatiotemporal / air-quality datasets.

Loads the index-windowed ``.npz`` datasets shipped by the CauAir repository
(https://github.com/PoorOtterBob/CauAir) and exposes them through TSFLab's
``(input_series, output_series, input_stamp, output_stamp)`` item contract.

On-disk layout (per dataset root, e.g. ``dataset/cauair_ccaq/``)::

    his.npz            # keys: data (T, N, C), mean (C,), std (C,)
    idx_train.npy      # window-centre indices into ``data``
    idx_val.npy
    idx_test.npy
    adj_mx.npy         # (N, N) adjacency (optional; graph models only)

Channel 0 of ``data`` is the target value; channels ``1:`` are per-node
covariates (meteorology, calendar features, ...). A window centred at ``i``
covers history ``[i - seq_len + 1, i]`` and future ``[i + 1, i + horizon]``,
matching CauAir's ``DataLoader`` offsets.

Two registered names share this class:

* ``cauair_st`` — spatiotemporal layout. ``input_series`` is the value
  ``(seq_len, N)``; the covariates ``(seq_len, N, F)`` ride in ``input_stamp``
  so the spatiotemporal / air-quality batch assembler can rebuild
  ``(B, T, N, 1 + F)`` (and the future covariate block).
* ``cauair_ts`` — plain time-series layout. The ``N`` node values become the
  ``C`` channels of a classic ``(seq_len, N)`` forecasting tensor; covariates
  are dropped and zero calendar stamps are returned.
"""

from __future__ import annotations

import os

import numpy as np
from torch.utils.data import Dataset

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.schemas.datasets.cauair import DatasetParameterConfig


class _CauAirBase(Dataset):
    """Index-windowed CauAir dataset.

    Parameters
    ----------
    root_path : str
        Directory containing ``his.npz`` and ``idx_*.npy``.
    data_path : str
        Unused (kept for the provider's call signature); pass ``""``.
    size : tuple[int, int, int]
        ``(seq_len, label_len, pred_len)``. ``pred_len`` is the horizon.
    flag : str
        Split: ``"train"`` | ``"val"`` | ``"test"``.
    features : str
        Unused; node layout is fixed by the data.
    input_dim : int
        Number of channels to keep from ``data`` (value + covariates).
    npz_name : str
        Name of the bundle file (default ``his.npz``).
    scale : bool
        Whether to z-score using the bundle's per-channel ``mean``/``std``.
    """

    spatiotemporal = True

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "M",
        input_dim: int = 8,
        npz_name: str = "his.npz",
        scale: bool = True,
        max_windows: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size
        self.input_dim = input_dim
        self.scale = scale
        self.max_windows = max_windows
        self._load(root_path, npz_name, flag, input_dim)

    def _load(self, root_path, npz_name, flag, input_dim):
        bundle = np.load(os.path.join(root_path, npz_name), allow_pickle=True)
        data = np.asarray(bundle["data"], dtype=np.float32)  # (T, N, C)
        if data.ndim != 3:
            raise ValueError(f"CauAir data must be (T, N, C); got {data.shape}")
        data = data[..., :input_dim]
        if self.scale and "mean" in bundle and "std" in bundle:
            mean = np.asarray(bundle["mean"], dtype=np.float32).reshape(-1)
            std = np.asarray(bundle["std"], dtype=np.float32).reshape(-1)
            # mean/std may be scalar (global) or per-channel (length C).
            if mean.size >= input_dim:
                mean, std = mean[:input_dim], std[:input_dim]
            std = np.where(std == 0, 1.0, std)
            data = (data - mean) / std
            # Statistic for the value channel (channel 0), used for inversion.
            self.value_mean = float(mean[0])
            self.value_std = float(std[0])
        else:
            self.value_mean = self.value_std = None
        self.data = data
        self.num_nodes = data.shape[1]
        # Optional adjacency matrix (N, N) for graph models. Exposed as
        # ``self.adj_mx`` so the runner can inject it into the model factory;
        # ``None`` when the bundle ships no ``adj_mx.npy`` (non-graph models).
        adj_path = os.path.join(root_path, "adj_mx.npy")
        if os.path.exists(adj_path):
            adj = np.asarray(np.load(adj_path, allow_pickle=True), dtype=np.float32)
            self.adj_mx = adj[: self.num_nodes, : self.num_nodes]
        else:
            self.adj_mx = None
        idx = np.load(os.path.join(root_path, f"idx_{flag}.npy"))
        idx = np.asarray(idx).reshape(-1).astype(np.int64)
        if self.max_windows is not None and len(idx) > self.max_windows:
            # Evenly subsample window centres (smoke runs / quick checks).
            sel = np.linspace(0, len(idx) - 1, self.max_windows).astype(np.int64)
            idx = idx[sel]
        self.idx = idx

    def __len__(self) -> int:
        """Number of valid windows in this split."""
        return len(self.idx)

    def _window(self, center: int):
        """Return history and future ``(T, N, C)`` slices for a window centre."""
        h0 = center - self.seq_len + 1
        hist = self.data[h0 : center + 1]  # (seq_len, N, C)
        fut = self.data[center + 1 : center + 1 + self.pred_len]  # (pred_len, N, C)
        return hist, fut

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse z-score the target channel (channel 0)."""
        if self.value_mean is None:
            return data
        return data * self.value_std + self.value_mean


class Dataset_CauAir_ST(_CauAirBase):
    """Spatiotemporal layout: value in ``input_series``, covariates in stamps."""

    spatiotemporal = True

    def __getitem__(self, index: int) -> tuple:
        """Return ``(value_hist, value_fut, cov_hist, cov_fut)``.

        Shapes: value ``(T, N)``; covariates ``(T, N, F)`` with
        ``F = input_dim - 1``. The covariates ride in the stamp slots so the
        spatiotemporal batch assembler can rebuild ``(B, T, N, 1 + F)``.
        """
        center = int(self.idx[index])
        hist, fut = self._window(center)
        value_hist = np.ascontiguousarray(hist[..., 0])  # (seq_len, N)
        value_fut = np.ascontiguousarray(fut[..., 0])  # (pred_len, N)
        cov_hist = np.ascontiguousarray(hist[..., 1:])  # (seq_len, N, F)
        cov_fut = np.ascontiguousarray(fut[..., 1:])  # (pred_len, N, F)
        return value_hist, value_fut, cov_hist, cov_fut


class Dataset_CauAir_TS(_CauAirBase):
    """Plain time-series layout: the ``N`` node values become the channels."""

    spatiotemporal = False

    def __getitem__(self, index: int) -> tuple:
        """Return ``(value_hist, value_fut, zero_stamp, zero_stamp)``.

        Value tensors are ``(T, N)`` (nodes as channels); covariates are
        dropped and zero calendar stamps ``(T, 6)`` are returned to match the
        forecasting item contract.
        """
        center = int(self.idx[index])
        hist, fut = self._window(center)
        value_hist = np.ascontiguousarray(hist[..., 0])  # (seq_len, N)
        value_fut = np.ascontiguousarray(fut[..., 0])  # (pred_len, N)
        in_stamp = np.zeros((value_hist.shape[0], 6), dtype=np.float32)
        out_stamp = np.zeros((value_fut.shape[0], 6), dtype=np.float32)
        return value_hist, value_fut, in_stamp, out_stamp


def register() -> None:
    """Register the CauAir spatiotemporal and time-series datasets."""
    DATASET_REGISTRY.register(
        "cauair_st",
        Dataset_CauAir_ST,
        DatasetParameterConfig,
        task_modes=frozenset({"spatiotemporal", "covariate"}),
        storage="directory",
    )
    DATASET_REGISTRY.register(
        "cauair_ts",
        Dataset_CauAir_TS,
        DatasetParameterConfig,
        task_modes=frozenset({"time_series"}),
        storage="directory",
    )
