"""Runtime specification for DPANet."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dpanet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_scales: int = Field(default=3, ge=1, le=8)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _heads(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="DPANet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dpanet.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
