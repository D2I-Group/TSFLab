"""Runtime specification for MAFS."""

from __future__ import annotations

from tsflab.experiments.runner.objective import TrainingBatch

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mafs.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    num_agents: int = 4
    num_layers: int = 2
    num_heads: int = 4
    topology: str = "star"


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        num_agents=params["num_agents"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        topology=params["topology"],
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion plus the multi-scale prefix specialization loss."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.specialization_loss(batch.x, batch.target)


SPEC = ModelSpec(
    name="MAFS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
