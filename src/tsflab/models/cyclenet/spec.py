"""Model specification for CycleNet."""

from __future__ import annotations

from typing import Literal

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cyclenet.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cycle: int = Field(default=24, gt=0)
    model_type: Literal["linear", "mlp"] = "linear"
    d_model: int = Field(default=512, gt=0)
    use_revin: bool = True


def build_model(cfg, params):
    """Construct CycleNet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        cycle=params["cycle"],
        model_type=params["model_type"],
        d_model=params["d_model"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="CycleNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
