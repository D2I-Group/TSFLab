"""Parameters for the synthetic polynomial-trend generator."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    target: str = "OT"
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
    channel_number: int = 1
    num_samples: int = 1024
    degree_min: int = 2
    degree_max: int = 6
    coeff_range: list[float] = Field(default_factory=lambda: [-0.8, 0.8])
    noise_std: float = 0.1
    normalize_t: bool = True
