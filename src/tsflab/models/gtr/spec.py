"""Runtime specification for GTR."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gtr.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    dropout: float = 0.1
    cycle_length: int = Field(default=168, ge=1)
    local_period: int = Field(default=24, ge=1)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        cycle_length=params["cycle_length"],
        local_period=params["local_period"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="GTR",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
