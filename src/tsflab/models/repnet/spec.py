"""Runtime specification for REPNet."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.repnet.model import Model

Representation = Literal[
    "single_linear",
    "linear",
    "glu",
    "simple_linear",
    "double_linear",
    "double_linear_glu",
    "cnn_1",
    "cnn_2",
    "cnn_3",
]


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_extractors: list[list[int]] = Field(
        default_factory=lambda: [[3, 1, 1], [10, 2, 4], [15, 3, 5]], min_length=1
    )
    encoding_size: int = Field(default=16, gt=0)
    representation: Representation = "linear"
    time_embedding: list[Literal["posEmb", "timeF", "tempEmb"]] = Field(
        default_factory=lambda: ["posEmb"]
    )
    time_embedding_size: int = Field(default=16, gt=0)
    freq: str = "h"
    memory_layers: int = Field(default=3, ge=0)
    attention: bool = False
    n_heads: int = Field(default=16, gt=0)
    glu: bool = True
    joint_feature_mixing: bool = False
    lstm_layers: int = Field(default=0, ge=0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    revin: bool = True

    @field_validator("patch_extractors")
    @classmethod
    def _triples(cls, value: list[list[int]]) -> list[list[int]]:
        if any(len(spec) != 3 or min(spec) < 1 for spec in value):
            raise ValueError(
                "each patch extractor is [cover_size, dilation, stride] of positive integers"
            )
        return value

    @field_validator("time_embedding")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("time_embedding entries must be unique")
        return value

    @model_validator(mode="after")
    def _widths(self):
        if self.time_embedding and self.encoding_size % 2:
            raise ValueError("encoding_size must be even when a time embedding is used")
        if "posEmb" in self.time_embedding and self.time_embedding_size % 2:
            raise ValueError("posEmb needs an even time_embedding_size")
        if self.attention and self.encoding_size % self.n_heads:
            raise ValueError("encoding_size must be divisible by n_heads when attention is enabled")
        return self


def build_model(cfg, params):
    if max(spec[0] for spec in params["patch_extractors"]) > cfg.task.seq_len:
        raise ValueError("REPNet patch extractor cover sizes must not exceed task.seq_len")
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="REPNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_repnet.toml",
    capabilities=frozenset({"time-series"}),
    components=("marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
