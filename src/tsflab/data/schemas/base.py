"""Strict base class for registered dataset parameter schemas."""

from pydantic import BaseModel, ConfigDict


class DatasetParameters(BaseModel):
    """Reject misspelled dataset options at configuration load time."""

    model_config = ConfigDict(extra="forbid")
