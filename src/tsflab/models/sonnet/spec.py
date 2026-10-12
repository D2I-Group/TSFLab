"""Model specification for Sonnet."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sonnet.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=16, gt=0)
    num_wavelets: int = Field(default=4, gt=0)
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct Sonnet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        num_wavelets=params["num_wavelets"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="Sonnet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
