"""Runtime specification for PRformer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.prformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    conv_windows: list[int] = Field(default_factory=lambda: [24, 48, 72, 96, 144], min_length=1)
    conv_channels: int = Field(default=128, gt=0)
    rnn_mix_temperature: float = Field(default=0.002, gt=0.0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    causal_variate_mask: bool = False

    @model_validator(mode="after")
    def _contract(self) -> ModelParameterConfig:
        windows = self.conv_windows
        if windows[0] < 1 or any(b <= a for a, b in zip(windows, windows[1:])):
            raise ValueError("conv_windows must be strictly increasing positive period lengths")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="PRformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_prformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 720, "pred_len": 96, "label_len": 0},
)
