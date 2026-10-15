"""Synthetic periodic dataset with independent short sequences."""

from __future__ import annotations

import numpy as np

from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.datasets.base import ForecastingDataset
from tsflab.data.schemas.datasets.periodic import DatasetParameterConfig


class Dataset_periodic(ForecastingDataset):
    """Synthetic periodic dataset with random sine waves per sample."""

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
        period: int = 24,
        noise_std: float = 0.1,
        amplitude_range: tuple[float, float] = (0.5, 1.5),
        phase_range: tuple[float, float] = (0.0, 2 * np.pi),
        cycle_start_mode: str = "random",
        random_phase: bool = True,
    ):
        self.channel_number = channel_number
        self.num_samples = num_samples
        self.period = period
        self.noise_std = noise_std
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.cycle_start_mode = cycle_start_mode
        self.random_phase = random_phase
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
        """Generate periodic sine data and return split series and timestamps."""
        total_len = self.seq_len + self.pred_len
        num_samples = self.num_samples

        t = np.arange(total_len, dtype=np.float32)
        t = t[None, :, None]

        if self.cycle_start_mode == "random":
            cycle_start = np.random.randint(0, self.period, size=num_samples)
        else:
            cycle_start = np.zeros(num_samples, dtype=np.int64)
        phase_offset = 2 * np.pi * cycle_start.astype(np.float32) / self.period
        phase_offset = phase_offset[:, None, None]

        amp_low, amp_high = self.amplitude_range
        amplitude = np.random.uniform(amp_low, amp_high, size=(num_samples, 1, self.channel_number))

        if self.random_phase:
            phase_low, phase_high = self.phase_range
            random_phase = np.random.uniform(phase_low, phase_high, size=(num_samples, 1, self.channel_number))
        else:
            random_phase = 0.0

        signal = np.sin(2 * np.pi * t / self.period + phase_offset + random_phase)
        noise = np.random.normal(0.0, self.noise_std, size=signal.shape)
        data = amplitude * signal + noise

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

        time_stamp = self._build_time_stamp_matrix(total_len, cycle_start)

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

    def _build_time_stamp_matrix(self, total_len: int, cycle_start: np.ndarray) -> np.ndarray:
        t = np.arange(total_len, dtype=np.int64)
        hour = (t[None, :] + cycle_start[:, None]) % self.period
        day = (t[None, :] // self.period).astype(np.int64)
        day = np.broadcast_to(day, hour.shape)
        weekday = day % 7

        year = np.zeros_like(hour)
        month = np.ones_like(hour)
        minute = np.zeros_like(hour)

        stamp = np.stack([year, month, day, weekday, hour, minute], axis=-1)
        return stamp.astype(np.int64)


def register() -> None:
    """Register the periodic dataset."""
    DATASET_REGISTRY.register(
        "periodic",
        Dataset_periodic,
        DatasetParameterConfig,
        task_modes=frozenset({"time_series"}),
    )
