"""Model specification for TimeBase."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timebase.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    period_len: int = Field(default=24, gt=0)
    basis_num: int = Field(default=6, gt=0)
    individual: bool = False
    orthogonal_weight: float = Field(default=0.08, ge=0.0)
    use_period_norm: bool = True


def build_model(cfg, params):
    """Construct TimeBase from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        period_len=params["period_len"],
        basis_num=params["basis_num"],
        individual=bool(params["individual"]),
        orthogonal_weight=float(params["orthogonal_weight"]),
        use_period_norm=bool(params["use_period_norm"]),
    )


SPEC = ModelSpec(
    name="TimeBase",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
