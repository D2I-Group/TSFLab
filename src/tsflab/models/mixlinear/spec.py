"""Model specification for MixLinear."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mixlinear.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    downsample: int = Field(default=4, gt=0)
    segments: int = Field(default=4, gt=0)
    hidden_rank: int = Field(default=2, gt=0)
    spectral_rank: int = Field(default=2, gt=0)


def build_model(cfg, params):
    """Construct MixLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        downsample=params["downsample"],
        segments=params["segments"],
        hidden_rank=params["hidden_rank"],
        spectral_rank=params["spectral_rank"],
    )


SPEC = ModelSpec(
    name="MixLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
