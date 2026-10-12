"""Runtime specification for PULSE."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pulse.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=32, gt=0)
    phase_period: int = Field(default=24, gt=0)
    phase_resolution: int = Field(default=8, gt=0)
    router_heads: int = Field(default=4, gt=0)
    dropout: float = 0.1
    eps: float = 1e-5

    @model_validator(mode="after")
    def _heads_divide_width(self):
        if self.d_model % self.router_heads:
            raise ValueError("d_model must be divisible by router_heads")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        phase_period=params["phase_period"],
        phase_resolution=params["phase_resolution"],
        router_heads=params["router_heads"],
        dropout=params["dropout"],
        eps=params["eps"],
    )


SPEC = ModelSpec(
    name="PULSE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
