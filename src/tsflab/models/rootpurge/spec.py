"""Runtime specification for RootPurge."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.rootpurge.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    domain: Literal["time", "frequency"] = "frequency"
    individual: bool = False
    instance_norm: bool = True
    purge_lambda: float = Field(default=0.25, ge=0)
    purge_order: int = Field(default=1, ge=1)


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (3): configured root-seeking loss plus the root-purging regularizer.

    The residual is taken on every observed channel of the horizon (for ``MS``
    the root-seeking term still scores only the target channel).
    """
    forecast = batch.forecast(model)
    seeking = criterion(batch.align(forecast), batch.target)
    residual = batch.y[:, -batch.pred_len :, :] - forecast[:, -batch.pred_len :, :]
    return forecast, seeking + model.purge_loss(residual)


SPEC = ModelSpec(
    name="RootPurge",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_rootpurge.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
