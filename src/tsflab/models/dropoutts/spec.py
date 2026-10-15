"""Runtime specification for DropoutTS."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dropoutts.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    padding_patch: Literal["end"] | None = "end"
    e_layers: int = Field(default=1, gt=0)
    d_model: int = Field(default=256, gt=0)
    n_heads: int = Field(default=1, gt=0)
    d_ff: int = Field(default=1024, gt=0)
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["BatchNorm", "LayerNorm"] = "BatchNorm"
    attn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    revin: bool = True
    affine: bool = True
    p_min: float = Field(default=0.05, ge=0.0, lt=1.0)
    p_max: float = Field(default=0.5, gt=0.0, lt=1.0)
    init_alpha: float = 10.0
    init_sensitivity: float = 1.0

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.p_min >= self.p_max:
            raise ValueError("p_min must be smaller than p_max")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="DropoutTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dropoutts.toml",
    capabilities=frozenset({"time-series"}),
    components=("patchtst",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
