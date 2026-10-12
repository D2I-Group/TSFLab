"""Runtime specification for GOTSF."""

from __future__ import annotations

from tsflab.experiments.runner.objective import TrainingBatch

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gotsf.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    dropout: float = 0.1
    num_intervals: int = Field(default=4, gt=0)
    interval_min: float = -2.0
    interval_max: float = 2.0
    decay_rate: float = Field(default=50.0, ge=0)
    classification_weight: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.interval_max <= self.interval_min:
            raise ValueError("interval_max must be larger than interval_min")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        num_intervals=params["num_intervals"],
        interval_min=params["interval_min"],
        interval_max=params["interval_max"],
        decay_rate=params["decay_rate"],
        classification_weight=params["classification_weight"],
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """GOTSF Eqs. (9)--(12) over every interval; the criterion is not used."""
    forecast, loss = model.goal_oriented_objective(batch.x, batch.target)
    return forecast, loss


SPEC = ModelSpec(
    name="GOTSF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
