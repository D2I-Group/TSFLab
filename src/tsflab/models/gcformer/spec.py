"""Runtime specification for GCformer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gcformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    context_len: int | None = Field(default=None, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_ff: int = Field(default=512, gt=0)
    e_layers: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    fc_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    individual: bool = True
    kernel_dim: int = Field(default=32, gt=0)
    kernel_decay: float = Field(default=2.0, gt=0.0)
    h_channel: int = Field(default=32, gt=0)
    h_token: int = Field(default=512, gt=0)
    decoder_attention: Literal["full", "prob"] = "full"
    atten_bias: float = Field(default=0.5, ge=0.0, le=1.0)
    tc_bias: float = Field(default=1.0, ge=0.0, le=1.0)
    global_bias: float = 0.5
    local_bias: float = 0.5
    norm_type: Literal["revin", "seq_last"] = "revin"

    @model_validator(mode="after")
    def _contract(self) -> ModelParameterConfig:
        if self.d_model % self.n_heads or self.h_channel % self.n_heads or self.h_token % self.n_heads:
            raise ValueError("d_model, h_channel, and h_token must be divisible by n_heads")
        if self.context_len is not None and self.patch_len > self.context_len:
            raise ValueError("patch_len must not exceed context_len")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="GCformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gcformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "positional_encoding", "revin", "self_attention_family"),
    contract_task={"seq_len": 104, "pred_len": 24, "label_len": 0},
)
