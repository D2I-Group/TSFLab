"""Model specification for DLinear."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dlinear.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False

    @field_validator("kernel_size")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return value


def build_model(cfg, params):
    """Construct DLinear from a validated run configuration."""
    return Model(
        c_in=params["enc_in"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        kernel_size=params["kernel_size"],
        individual=params["individual"],
    )


SPEC = ModelSpec(
    name="DLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
