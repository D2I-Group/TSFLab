"""Runtime specification for Ister."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.ister.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, ge=2)
    d_ff: int = Field(default=2048, gt=0)
    layers: int = Field(default=2, gt=0)
    moving_avg: int = Field(default=25, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    use_marks: bool = True
    freq: Literal["h", "t", "s", "m", "a", "w", "d", "b"] = "h"
    recon_len: int = Field(default=48, ge=0)
    gelu_scores: bool = True

    @model_validator(mode="after")
    def _contract(self) -> ModelParameterConfig:
        if self.moving_avg % 2 == 0:
            raise ValueError("moving_avg must be odd (edge-padded moving average keeps the length)")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    if options["recon_len"] > cfg.task.seq_len:
        raise ValueError("recon_len must not exceed seq_len")
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_objective(model, batch: TrainingBatch, criterion):
    """Official non-AMP training loss: the configured criterion over the
    reconstruction of the last ``recon_len`` history steps and the horizon
    (``recon_len = 0`` gives the paper's horizon-only L2 when the criterion is MSE)."""
    full = model.full_forecast(batch.x, batch.x_mark)
    forecast = full[:, model.recon_len :, :]
    if model.recon_len == 0:
        return forecast, criterion(batch.align(forecast), batch.target)
    start = -1 if batch.features == "MS" else 0
    outputs = torch.cat((full[:, : model.recon_len, start:], batch.align(forecast)), dim=1)
    target = torch.cat((batch.x[:, -model.recon_len :, start:], batch.target), dim=1)
    return forecast, criterion(outputs, target)


SPEC = ModelSpec(
    name="Ister",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_ister.toml",
    capabilities=frozenset({"time-series"}),
    components=(
        "marks",
        "revin",
        "self_attention_family",
        "series_decomposition",
        "transformer_encdec",
    ),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
