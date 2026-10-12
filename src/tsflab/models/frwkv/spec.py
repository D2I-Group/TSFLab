"""Runtime specification for FRWKV."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.frwkv.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``weighted_l1_loss`` trains with the official horizon-weighted L1 objective
    ``mean(|y_hat - y| * (h + 1) ** -loss_alpha)``; set it to false to use the
    configured criterion instead.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    d_ff: int = Field(default=512, ge=2)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    embed_size: int = Field(default=16, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    weighted_l1_loss: bool = True
    loss_alpha: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def _heads_divide_width(self):
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


def horizon_weighted_l1(
    prediction: torch.Tensor, target: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Official ``WeightedL1Loss``: L1 error weighted by ``(h + 1) ** -alpha`` along the horizon."""
    steps = torch.arange(
        1, prediction.shape[1] + 1, device=prediction.device, dtype=prediction.dtype
    )
    weights = steps.pow(-alpha).view(1, -1, 1)
    return ((prediction - target).abs() * weights).mean()


def training_objective(model, batch, criterion):
    """Official horizon-weighted L1 while training (validation keeps the configured criterion)."""
    forecast = batch.forecast(model)
    prediction = batch.align(forecast)
    if model.weighted_l1_loss:
        return forecast, horizon_weighted_l1(prediction, batch.target, model.loss_alpha)
    return forecast, criterion(prediction, batch.target)


SPEC = ModelSpec(
    name="FRWKV",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_frwkv.toml",
    capabilities=frozenset({"time-series"}),
    components=("frwkv_linear_attention", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
