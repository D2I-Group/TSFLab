"""Synthetic trend dataset with polynomial sequences."""

from __future__ import annotations

import numpy as np

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.datasets.base import ForecastingDataset
from tsflab.data.schemas.datasets.trend import DatasetParameterConfig


class Dataset_trend(ForecastingDataset):
    """Synthetic trend dataset with random polynomial trends per sample."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
        scale: bool = True,
        channel_number: int = 1,
        num_samples: int = 1024,
        degree_min: int = 2,
        degree_max: int = 6,
        coeff_range: tuple[float, float] = (-0.8, 0.8),
        noise_std: float = 0.1,
        normalize_t: bool = True,
    ):
        self.channel_number = channel_number
        self.num_samples = num_samples
        self.degree_min = degree_min
        self.degree_max = degree_max
        self.coeff_range = coeff_range
        self.noise_std = noise_std
        self.normalize_t = normalize_t
        self.data_min = None
        self.data_max = None
        super().__init__(
            root_path=root_path,
            data_path=data_path,
            size=size,
            flag=flag,
            features=features,
            target=target,
            split_ratio=split_ratio,
            scale=scale,
        )

    def __len__(self) -> int:
        """Return number of independent samples in the split."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """Return one sample window and optional timestamps."""
        sample = self.data[index]
        input_series = sample[: self.seq_len]
        output_series = sample[self.seq_len - self.label_len : self.seq_len + self.pred_len]

        if self.time_stamp is not None:
            stamp = self.time_stamp[index]
            input_stamp = stamp[: self.seq_len]
            output_stamp = stamp[self.seq_len - self.label_len : self.seq_len + self.pred_len]
        else:
            input_stamp, output_stamp = None, None

        return input_series, output_series, input_stamp, output_stamp

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform min-max scaled data."""
        if self.data_min is None or self.data_max is None:
            return data
        scale = self.data_max - self.data_min
        scale = np.where(scale == 0, 1.0, scale)
        return (data + 1.0) / 2.0 * scale + self.data_min

    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate polynomial trend data and return split series and timestamps."""
        total_len = self.seq_len + self.pred_len
        num_samples = self.num_samples

        t = np.arange(total_len, dtype=np.float32)
        if self.normalize_t:
            denom = total_len - 1
            if denom <= 0:
                t = np.zeros_like(t)
            else:
                t = (t / denom) * 2.0 - 1.0

        degree_max = max(1, int(self.degree_max))
        degree_min = max(1, int(self.degree_min))
        if degree_min > degree_max:
            degree_min = degree_max
        t_powers = np.stack([np.power(t, k) for k in range(degree_max + 1)], axis=0)

        degrees = np.random.randint(degree_min, degree_max + 1, size=(num_samples, self.channel_number))
        coeff_low, coeff_high = self.coeff_range
        coeffs = np.random.uniform(
            coeff_low,
            coeff_high,
            size=(num_samples, self.channel_number, degree_max + 1),
        ).astype(np.float32)
        mask = np.arange(degree_max + 1)[None, None, :] <= degrees[:, :, None]
        coeffs = coeffs * mask

        trend = np.einsum("ncd,dt->nct", coeffs, t_powers).astype(np.float32)
        trend = np.transpose(trend, (0, 2, 1))

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=trend.shape).astype(np.float32)
            data = trend + noise
        else:
            data = trend

        if scale:
            total_ratio = sum(split_ratio)
            train_end = int(split_ratio[0] / total_ratio * num_samples)
            train_data = data[:train_end]
            data_min = train_data.min()
            data_max = train_data.max()
            scale_range = data_max - data_min
            if scale_range == 0:
                scale_range = 1.0
            data = (data - data_min) / scale_range
            data = data * 2.0 - 1.0
            self.data_min = data_min
            self.data_max = data_max

        time_stamp = np.zeros((num_samples, total_len, 6), dtype=np.int64)

        border1, border2 = self._get_sample_borders(flag, split_ratio, num_samples)
        series_data = data[border1:border2]
        time_stamp = time_stamp[border1:border2]

        if features in {"M", "MS"}:
            return series_data, time_stamp

        channel_index = 0
        if target != "OT":
            try:
                channel_index = int(target)
            except ValueError:
                channel_index = 0
        series_data = series_data[:, :, channel_index : channel_index + 1]
        return series_data, time_stamp

    def _get_sample_borders(
        self,
        flag: str,
        split_ratio: tuple[float, float, float],
        num_samples: int,
    ) -> tuple[int, int]:
        flag_map = {"train": 0, "val": 1, "test": 2}
        idx = flag_map[flag]
        total_ratio = sum(split_ratio)
        cum_ratios = [sum(split_ratio[: i + 1]) / total_ratio for i in range(len(split_ratio))]
        border1 = int(cum_ratios[idx - 1] * num_samples) if idx > 0 else 0
        border2 = int(cum_ratios[idx] * num_samples)
        return border1, border2


def register() -> None:
    """Register the trend dataset."""
    DATASET_REGISTRY.register(
        "trend",
        Dataset_trend,
        DatasetParameterConfig,
        task_modes=frozenset({"time_series"}),
    )
