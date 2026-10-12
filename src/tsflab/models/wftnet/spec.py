"""Runtime specification for WFTNet."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wftnet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    period_coeff: float = Field(ge=0.0, le=1.0)
    d_model: int = Field(default=512, gt=0, multiple_of=2)
    d_ff: int = Field(default=2048, gt=0)
    e_layers: int = Field(default=2, gt=0)
    top_k: int = Field(default=5, gt=0)
    num_kernels: int = Field(default=6, gt=0)
    wavelet_scale: int = Field(default=1, ge=0, le=12)
    pwc_power: int = Field(default=10, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="WFTNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_wftnet.toml",
    capabilities=frozenset({"time-series"}),
    components=("dominant_periods", "embed", "inception_block", "marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
