"""Runtime specification for STHD."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.sthd.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    k: int = Field(default=13, ge=0)
    patch_len: int = Field(default=12, ge=1)
    stride: int = Field(default=6, ge=1)
    d_model: int = Field(default=256, ge=1)
    n_heads: int = Field(default=4, ge=1)
    e_layers: int = Field(default=2, ge=1)
    d_ff: int = Field(default=384, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"

    @model_validator(mode="after")
    def check_heads(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


def _full_training_series(train_loader, enc_in: int):
    """Return the full scaled ``(T, C)`` training series behind the loader, or ``None``."""
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        return None
    series = torch.as_tensor(data).float()
    if series.ndim != 2 or series.shape[1] != enc_in or series.shape[0] < 2:
        return None
    return series


def training_setup(model, train_loader, *, pred_len, features):
    """Relation Matrix Sparsity (Sec. 4.1) on the training split before the first epoch.

    The official pipeline correlates the whole training series, so the loader's
    dataset series is used when available; otherwise the newest history step of
    every stride-1 training window (the training series minus the first lookback).
    """
    del pred_len, features
    series = _full_training_series(train_loader, model.enc_in)
    if series is None:
        steps = [batch_x[:, -1, :].float() for batch_x, *_ in train_loader]
        series = torch.cat(steps)
    model.fit_relations(series)


def training_objective(model, batch: TrainingBatch, criterion):
    """The configured observation loss on every target's forecast (the paper uses MSE)."""
    forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="STHD",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_sthd.toml",
    capabilities=frozenset({"time-series"}),
    components=(
        "embed",
        "flatten_forecast_head",
        "revin",
        "self_attention_family",
        "transformer_encdec",
    ),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
