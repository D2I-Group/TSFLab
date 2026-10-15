"""Runtime specification for SARAF."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.saraf.model import Model, series_from_windows


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    top_k: int = Field(default=20, gt=0)
    candidate_pool: int = Field(default=100, gt=0)
    time_aware_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sigma_min: float = Field(default=0.05, gt=0.0)
    sigma_max: float = Field(default=0.3, gt=0.0)
    lambda_min: float = Field(default=0.3, ge=0.0, le=1.0)
    lambda_max: float = Field(default=0.9, ge=0.0, le=1.0)
    mmr_temperature: float = Field(default=0.1, gt=0.0)
    n_sub_windows: int = Field(default=6, ge=2)
    freq: Literal["h", "t", "15min", "30min", "d", "w", "m", "s"] = "h"
    eval_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _contract(self) -> ModelParameterConfig:
        if self.top_k > self.candidate_pool:
            raise ValueError("top_k must not exceed candidate_pool")
        if self.sigma_min > self.sigma_max or self.lambda_min > self.lambda_max:
            raise ValueError("sigma_min <= sigma_max and lambda_min <= lambda_max are required")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_setup(model, train_loader, *, pred_len, features):
    """Build the retrieval database from the training split before the first epoch."""
    del features
    series, marks = series_from_windows(train_loader.dataset, pred_len)
    model.fit_database(series, marks)


def training_objective(model, batch: TrainingBatch, criterion):
    """The configured observation loss on the forecast (the paper trains with MSE)."""
    forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="SARAF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_saraf.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 96, "pred_len": 24, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
