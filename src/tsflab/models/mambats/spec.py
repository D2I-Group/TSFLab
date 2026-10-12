"""Runtime specification for MambaTS."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mambats.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    e_layers: int = Field(default=4, gt=0)
    d_state: int = Field(default=16, gt=0)
    d_conv: int = Field(default=4, gt=0)
    expand: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    patch_len: int = Field(default=48, gt=0)
    stride: int = Field(default=48, gt=0)
    use_causal_conv: bool = False
    vpt_mode: Literal[0, 1] = 1
    atsp_solver: Literal["SA", "GD"] = "SA"
    atsp_steps: int = Field(default=20000, ge=0)
    atsp_seed: int = 0


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **params,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Configured criterion, plus crediting each sample's loss to its scan transitions.

    The loss is the configured criterion on the whole batch (the paper's MSE when
    ``loss = "mse"``); the per-sample criterion values only feed the variable-pair
    cost matrix that VAST uses to choose the inference-time scan order.
    """
    forecast = batch.forecast(model)
    prediction, target = batch.align(forecast), batch.target
    loss = criterion(prediction, target)
    if model.vpt_mode == 1:
        with torch.no_grad():
            per_sample = torch.stack(
                [
                    criterion(prediction[i : i + 1], target[i : i + 1])
                    for i in range(prediction.shape[0])
                ]
            )
        model.update_scan_costs(per_sample)
    return forecast, loss


SPEC = ModelSpec(
    name="MambaTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mambats.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "mamba", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
