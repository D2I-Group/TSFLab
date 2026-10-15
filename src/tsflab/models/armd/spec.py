"""Runtime specification for ARMD."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.armd.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    sampling_steps: int = Field(default=1, gt=0)
    loss_type: Literal["l1", "l2"] = "l1"
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    w_grad: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **params,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Replace the configured loss by the paper's evolution-trend objective (Eq. 7).

    The objective needs the true future and the history jointly, over every channel,
    and does not produce a forecast from the same pass.
    """
    future = batch.y[:, -batch.pred_len :, :]
    return None, model.training_loss(batch.x, future)


SPEC = ModelSpec(
    name="ARMD",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_armd.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
