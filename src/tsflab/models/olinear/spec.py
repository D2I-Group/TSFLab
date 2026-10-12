"""Model specification for OLinear."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.olinear.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    d_model: int = Field(default=32, ge=1)
    dropout: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct OLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="OLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
