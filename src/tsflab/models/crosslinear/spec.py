"""Model specification for CrossLinear."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.crosslinear.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    d_model: int = Field(default=32, gt=0)
    d_ff: int = Field(default=128, gt=0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.5, ge=0.0, le=1.0)


def build_model(cfg, params):
    """Construct CrossLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params["patch_len"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        alpha=params["alpha"],
        beta=params["beta"],
    )


SPEC = ModelSpec(
    name="CrossLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
