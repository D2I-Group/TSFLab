"""Parameters for user-provided CSV forecasting datasets."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
    # Opt-in scaling controls (default off == previous behavior).
    target_channel: int | None = None
    norm_each_channel: bool = False
    # Values that mark missing readings (for example -9999). They become NaN and
    # are imputed causally (forward fill; a leading gap becomes zero) before
    # scaling. Empty keeps the file as-is.
    missing_sentinels: list[float] = Field(default_factory=list)


class WeatherParameterConfig(DatasetParameterConfig):
    """Native Weather defaults match its supplied preset; custom CSVs stay opt-in."""

    missing_sentinels: list[float] = Field(default_factory=lambda: [-9999.0])
