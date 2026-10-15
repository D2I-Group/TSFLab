"""Model specification for PAttn."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pattn.model import Model


class ModelParameterConfig(BaseModel):
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _heads(self) -> ModelParameterConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    """Construct PAttn from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="PAttn",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("instance_norm",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
