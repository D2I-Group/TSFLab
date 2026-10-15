"""Parameters for fixed-split ETT hourly and minute datasets."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [12.0, 4.0, 4.0])
    # Opt-in scaling controls (default off == previous behavior).
    target_channel: int | None = None
    norm_each_channel: bool = False
