"""Model specification for RLinear."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.rlinear.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    individual: bool = False
    affine: bool = False
    subtract_last: bool = False
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


def build_model(cfg, params):
    """Construct RLinear from a validated run configuration."""
    return Model(
        c_in=params["enc_in"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        individual=bool(params["individual"]),
        affine=bool(params["affine"]),
        subtract_last=bool(params["subtract_last"]),
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="RLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
