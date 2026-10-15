"""Named dataset selection, one readable path, and loader parameters."""

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    alias: str | None = None
    # TSFLab Leaderboard track for runs on this dataset. Defaults to the task
    # mode (time_series / spatiotemporal / covariate) when unset; set it to
    # "realtime" for periodically-refreshed live datasets (e.g. stock_hs300).
    track: str | None = None
    path: str = ""
    id: str | None = None
    params: SerializeAsAny[dict | BaseModel] = Field(default_factory=dict)
