"""Runtime specification for Dozerformer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dozerformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    label_len: int = Field(default=96, ge=0)
    patch_size: int = Field(default=24, gt=0)
    embed_dim: int = Field(default=8, gt=0)
    d_ff: int = Field(default=32, gt=0)
    n_heads: int = Field(default=4, gt=0)
    e_layers: int = Field(default=2, ge=1)
    d_layers: int = Field(default=1, ge=1)
    local_window: int = Field(default=3, ge=0)
    stride: int = Field(default=7, ge=0)
    vary_len: int = Field(default=1, ge=0)
    moving_avg: list[int] = Field(default_factory=lambda: [13, 17], min_length=1)
    trend_branch: bool = True
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    score_mode: Literal["zero", "masked"] = "zero"
    normalize_decoder_input: bool = False

    @field_validator("moving_avg")
    @classmethod
    def _odd_kernels(cls, value: list[int]) -> list[int]:
        if any(k < 1 or k % 2 == 0 for k in value):
            raise ValueError("moving_avg kernels must be positive odd integers")
        return value

    @model_validator(mode="after")
    def _heads_divide_width(self):
        if (self.embed_dim * self.patch_size) % self.n_heads:
            raise ValueError("embed_dim * patch_size must be divisible by n_heads")
        return self


def build_model(cfg, params):
    if params["label_len"] > cfg.task.seq_len:
        raise ValueError("Dozerformer label_len must not exceed task.seq_len")
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="Dozerformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_dozerformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
