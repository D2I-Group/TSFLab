"""Model specification for xPatch."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.xpatch.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    padding_patch: Literal["end", "none"] = "end"
    ma_type: Literal["ema", "dema"] = "ema"
    alpha: float = Field(default=0.3, gt=0.0, lt=1.0)
    beta: float = Field(default=0.3, gt=0.0, lt=1.0)
    revin: bool = True
    hidden_dim: int = Field(default=64, gt=0)


def build_model(cfg, params):
    """Construct xPatch from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        padding_patch=params["padding_patch"],
        ma_type=params["ma_type"],
        alpha=params["alpha"],
        beta=params["beta"],
        revin=bool(params["revin"]),
        hidden_dim=params["hidden_dim"],
    )


SPEC = ModelSpec(
    name="xPatch",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
