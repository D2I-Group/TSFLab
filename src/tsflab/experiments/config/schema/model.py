"""Named model selection and model-specific parameter mapping."""

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict = Field(default_factory=dict)
