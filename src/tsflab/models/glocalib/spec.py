"""Model specification for GlocalIB."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.glocalib.model import Model

from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Parameters for the Glocal-IB forecasting model."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    # Alignment regularizer weight and augmentation strength.
    align_weight: float = Field(default=0.5, ge=0.0)
    mask_ratio: float = Field(default=0.25, ge=0.0, lt=1.0)
    # "cos_align" (robust) or "contrastive" (InfoNCE over time steps).
    align_loss_type: Literal["cos_align", "contrastive"] = "cos_align"
    kl_weight: float = Field(default=0.01, ge=0.0)


def build_model(cfg, params):
    """Construct GlocalIB from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        align_weight=params["align_weight"],
        mask_ratio=params["mask_ratio"],
        align_loss_type=params["align_loss_type"],
        kl_weight=params["kl_weight"],
    )


SPEC = ModelSpec(
    name="GlocalIB",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_glocalib.toml",
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
