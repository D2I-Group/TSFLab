"""Parameters for the synthetic periodic-series generator."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    target: str = "OT"
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
    channel_number: int = 1
    num_samples: int = 1024
    period: int = 24
    noise_std: float = 0.1
    amplitude_range: list[float] = Field(default_factory=lambda: [0.5, 1.5])
    phase_range: list[float] = Field(default_factory=lambda: [0.0, 6.283185307179586])
    cycle_start_mode: str = "random"
    random_phase: bool = True
