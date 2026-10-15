"""Construction and two-stage runtime contract for StateFlow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stateflow.model import Model


class ModelParameterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enc_in: int = Field(gt=0)
    encoder_epochs: int = Field(default=10, ge=1)
    encoder_lr: float = Field(default=0.001, gt=0)
    hidden_size: Literal[16, 32, 64] = 32
    memory_size: Literal[8, 16, 32] = 16
    chunk_width: Literal[3, 5, 7] = 5
    chunk_stride: Literal[1, 2, 4] = 2
    instance_norm: Literal[True] = True


def build_model(cfg, params):
    if cfg.task.seq_len < max(6, params["chunk_width"] + 1):
        raise ValueError("StateFlow needs seq_len >= 6 and seq_len > chunk_width")
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        encoder_epochs=params["encoder_epochs"],
        encoder_lr=params["encoder_lr"],
        hidden_size=params["hidden_size"],
        memory_size=params["memory_size"],
        chunk_width=params["chunk_width"],
        chunk_stride=params["chunk_stride"],
    )


SPEC = ModelSpec(
    name="StateFlow",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_stateflow.toml",
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
