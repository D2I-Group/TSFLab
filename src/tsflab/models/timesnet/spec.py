"""Model specification for TimesNet."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timesnet.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    top_k: int = Field(default=5, gt=0)
    num_kernels: int = Field(default=6, gt=0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> ModelParameterConfig:
        if self.enc_in != self.c_out:
            raise ValueError("enc_in and c_out must match")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        label_len=cfg.task.label_len,
        pred_len=cfg.task.pred_len,
        **params,
    )


SPEC = ModelSpec(
    name="TimesNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("dominant_periods", "forecast_embedding", "inception_block", "instance_norm"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
