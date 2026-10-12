"""Runtime specification for CDPM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.cdpm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    diffusion_steps: int = Field(default=50, gt=0)
    schedule: Literal["cosine", "linear", "exponential", "inverse_sqrt"] = "cosine"
    mlp_hidden: int = Field(default=256, gt=0)
    emb_dim: int = Field(default=256, ge=4, multiple_of=2)
    patch_len: int = Field(default=8, gt=0)
    n_layers: int = Field(default=14, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    moving_avg: int = Field(default=5, gt=0)
    clip_denoised: bool = True
    eval_seed: int | None = Field(default=0, ge=0)


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    if options["moving_avg"] % 2 == 0:
        raise ValueError("moving_avg must be odd")
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (16): the observation loss on the teacher-forced forecast (noised true seasonal future).

    The full-channel future comes from ``batch.y``; the loss reads the configured target
    channels. No sampled forecast is produced in this pass.
    """
    future = batch.y[:, -batch.pred_len :, :]
    prediction = model.teacher_forced_forecast(batch.x, future)
    return None, criterion(batch.align(prediction), batch.target)


SPEC = ModelSpec(
    name="CDPM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_cdpm.toml",
    capabilities=frozenset({"time-series"}),
    components=("series_decomposition",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
