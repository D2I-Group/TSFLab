"""Runtime specification for CLoRA."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.clora.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    backbone: Literal["itransformer", "rmlp"] = "itransformer"
    d_model: int = Field(default=512, gt=1)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=512, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    rank: int = Field(default=16, gt=0)
    adaptation_dim: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.adaptation_dim >= self.d_model:
            raise ValueError("adaptation_dim must be smaller than d_model")
        if self.backbone == "itransformer" and self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="CLoRA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_clora.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "revin", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
