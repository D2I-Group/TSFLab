"""Model specification for FEDformer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.fedformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    dec_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    moving_avg: int = Field(default=25, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    mode_select: Literal["random", "low"] = "random"
    modes: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> ModelParameterConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.moving_avg % 2 == 0:
            raise ValueError("moving_avg must be odd")
        if not (self.enc_in == self.dec_in == self.c_out):
            raise ValueError("enc_in, dec_in, and c_out must match")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        label_len=cfg.task.label_len,
        pred_len=cfg.task.pred_len,
        **params,
    )


SPEC = ModelSpec(
    name="FEDformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("decomposition_encdec", "forecast_embedding", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
