"""Model specification for SVTime."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.svtime.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    period: int = Field(default=24, gt=0)
    patch_size: int = Field(default=6, gt=0)
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False


def build_model(cfg, params):
    """Construct SVTime from a validated run configuration."""
    return Model(
        c_in=params["enc_in"],
        period=params["period"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        patch_size=params["patch_size"],
        revin=bool(params["revin"]),
        affine=bool(params["affine"]),
        subtract_last=bool(params["subtract_last"]),
    )


SPEC = ModelSpec(
    name="SVTime",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
