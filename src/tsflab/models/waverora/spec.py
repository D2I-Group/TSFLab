"""Runtime specification for WaveRoRA."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.waverora.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    wavelet: Literal["sym3", "coif3", "haar"] = "sym3"
    wavelet_layers: int = Field(default=3, ge=1, le=8)
    wavelet_dim: int = Field(default=64, gt=0)
    d_model: int = Field(default=256, gt=0)
    d_ff: int = Field(default=256, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=1, gt=0)
    router_num: int = Field(default=0, ge=0)
    kernel_size: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    rotary: bool = True
    residual: bool = True
    gate: bool = True

    @model_validator(mode="after")
    def check_shapes(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="WaveRoRA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_waverora.toml",
    capabilities=frozenset({"time-series"}),
    components=("orthogonal_dwt", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
