"""Model specification for DLinear."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    kernel_size: int = 25
    individual: bool = False


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
