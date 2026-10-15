"""Model specification for PatchTST."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.patchtst.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    padding_patch: Literal["end", "none"] = "end"
    e_layers: int = Field(default=3, gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_k: int | None = Field(default=None, gt=0)
    d_v: int | None = Field(default=None, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["BatchNorm", "LayerNorm"] = "BatchNorm"
    attn_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    ffn_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    res_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    proj_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    pre_norm: bool = False
    pe: Literal["zeros", "sincos"] = "zeros"
    learn_pe: bool = False
    individual: bool = False
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False

    @model_validator(mode="after")
    def _architecture_contract(self) -> ModelParameterConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        width = self.d_model // self.n_heads
        if self.d_k not in {None, width} or self.d_v not in {None, width}:
            raise ValueError("d_k and d_v must be omitted or equal d_model / n_heads")
        return self


def build_model(cfg, params):
    return Model(
        c_in=params.pop("enc_in"),
        context_window=cfg.task.seq_len,
        target_window=cfg.task.pred_len,
        n_layers=params.pop("e_layers"),
        **params,
    )


SPEC = ModelSpec(
    name="PatchTST",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
