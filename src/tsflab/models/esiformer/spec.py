"""Runtime specification for Esiformer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.esiformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    interpolation: Literal["none", "two_aver", "four_aver"] = "two_aver"
    sparse_ffn: bool = True
    sparsity: float = Field(default=0.3, ge=0.0, lt=1.0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    use_norm: bool = True
    use_marks: bool = True
    freq: Literal["h"] = "h"

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="Esiformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_esiformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "marks", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
