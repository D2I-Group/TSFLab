"""Parameter schema for the synthetic spatiotemporal dataset."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    """Validated parameters for the ``synthetic_st`` dataset."""

    num_nodes: int = 8
    steps_per_day: int = 24
    length: int = 600
    split_ratio: list[float] = Field(default_factory=lambda: [0.6, 0.2, 0.2])
    scale: bool = True
