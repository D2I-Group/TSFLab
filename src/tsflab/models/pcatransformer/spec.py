"""Runtime specification for PCATransformer."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.pcatransformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=2)
    n_components: int = Field(default=2, ge=1)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    embed: Literal["timeF", "fixed", "learned"] = "timeF"
    freq: str = "h"

    @model_validator(mode="after")
    def _components_fit_covariates(self) -> "ModelParameterConfig":
        if self.n_components > self.enc_in - 1:
            raise ValueError(f"n_components must be in [1, enc_in - 1 = {self.enc_in - 1}]")
        return self


def build_model(cfg, params):
    if cfg.task.features != "MS":
        raise ValueError(
            "PCATransformer reduces covariates and forecasts the final target channel; "
            "it requires task.features = 'MS'"
        )
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        enc_in=params["enc_in"],
        n_components=params["n_components"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        activation=params["activation"],
        embed=params["embed"],
        freq=params["freq"],
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
    """Fit the covariate PCA on the scaled training split before the first epoch.

    The full training series behind the loader is used when available; otherwise the
    newest history step of every training window (stride-1 windows cover the training
    series minus the first lookback).
    """
    del pred_len
    if features != "MS":
        raise ValueError("PCATransformer requires features = 'MS'")
    series = _full_training_series(train_loader, model.enc_in)
    if series is None:
        steps = [batch_x[:, -1, :].float() for batch_x, *_ in train_loader]
        series = torch.cat(steps)
    model.fit_pca(series.to(model.pca_mean.device))


def training_objective(model, batch: TrainingBatch, criterion):
    """The configured observation loss on the target forecast (the paper trains with MSE)."""
    forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="PCATransformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_pcatransformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 48, "features": "MS"},
    training_objective=training_objective,
    training_setup=training_setup,
)
