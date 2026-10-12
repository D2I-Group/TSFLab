"""Model specification for NLinear."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.nlinear.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    individual: bool = False


def build_model(cfg, params):
    """Construct NLinear from a validated run configuration."""
    return Model(
        c_in=params["enc_in"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        individual=bool(params["individual"]),
    )


SPEC = ModelSpec(
    name="NLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "last_value_center"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
