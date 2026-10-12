"""Model specification for SparseTSF."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sparsetsf.model import Model

from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    period: int = Field(default=24, gt=0)
    d_model: int = Field(default=64, gt=0)
    model_type: Literal["linear", "mlp"] = "linear"


def build_model(cfg, params):
    """Construct SparseTSF from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        period=params["period"],
        d_model=params["d_model"],
        model_type=params["model_type"],
    )


SPEC = ModelSpec(
    name="SparseTSF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
