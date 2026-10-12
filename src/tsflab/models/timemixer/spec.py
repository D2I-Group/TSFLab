"""Model specification for TimeMixer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timemixer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_model: int = Field(default=512, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    down_sampling_window: int = Field(default=2, ge=2)
    down_sampling_layers: int = Field(default=2, ge=1)
    moving_avg: int = Field(default=25, gt=0)
    top_k: int = Field(default=5, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    use_norm: bool = True
    decomp_method: Literal["moving_avg", "dft_decomp"] = "moving_avg"

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.enc_in != self.c_out:
            raise ValueError("enc_in and c_out must match")
        if self.moving_avg % 2 == 0:
            raise ValueError("moving_avg must be odd")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="TimeMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
