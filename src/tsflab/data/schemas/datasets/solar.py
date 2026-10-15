"""Parameters for solar-power forecasting CSV datasets."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
    # Opt-in scaling controls (default off == previous behavior).
    target_channel: int | None = None
    norm_each_channel: bool = False
    # Opt-in synthetic timestamps: the file has no date column. When ``start`` is
    # set, row i is stamped ``start + i * freq`` (pandas offset alias).
    start: str | None = None
    freq: str = "10min"
