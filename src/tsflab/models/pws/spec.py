"""Model specification for PWS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pws.model import Model

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    period: int = Field(default=24, gt=0)
    patch_size: int = Field(default=6, gt=0)
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    analysis_act: Literal["relu", "gelu", "silu", "tanh", "leaky_relu"] = "relu"
    analysis_hidden: list[int] = Field(default_factory=lambda: [512, 256])

    @field_validator("analysis_hidden")
    @classmethod
    def _positive_hidden_sizes(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("analysis_hidden entries must be positive")
        return values


def build_model(cfg, params):
    """Construct PWS from a validated run configuration."""
    return Model(
        c_in=params["enc_in"],
        period=params["period"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        patch_size=params["patch_size"],
        revin=bool(params["revin"]),
        affine=bool(params["affine"]),
        subtract_last=bool(params["subtract_last"]),
        analysis_act=params["analysis_act"],
        analysis_hidden=params["analysis_hidden"],
    )


SPEC = ModelSpec(
    name="PWS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
