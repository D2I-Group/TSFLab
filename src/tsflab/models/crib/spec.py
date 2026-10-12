"""Runtime specification for CRIB."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.crib.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=8, gt=0)
    model_dim: int = Field(default=32, gt=0)
    heads_num: int = Field(default=4, gt=0)
    enc_num: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["relu", "gelu"] = "relu"
    consis_weight: float = Field(default=1.0, ge=0.0)
    kl_weight: float = Field(default=1e-6, ge=0.0)
    augmentation_rate: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.model_dim % self.heads_num:
            raise ValueError("model_dim must be divisible by heads_num")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="CRIB",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_crib.toml",
    capabilities=frozenset(["time-series", "missing-values"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
