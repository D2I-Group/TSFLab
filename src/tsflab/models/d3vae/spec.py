"""Runtime specification for D3VAE."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.d3vae.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    freq: Literal["h", "t"] = "h"
    embedding_dim: int = Field(default=64, gt=0, multiple_of=2)
    hidden_size: int = Field(default=128, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    diff_steps: int = Field(default=100, gt=0)
    beta_start: float = Field(default=0.0, ge=0.0, lt=1.0)
    beta_end: float = Field(default=0.01, ge=0.0, lt=1.0)
    target_scale: float = Field(default=0.1, gt=0.0, lt=1.0)
    num_channels_enc: int = Field(default=32, gt=0)
    num_channels_dec: int = Field(default=32, gt=0)
    num_blocks: int = Field(default=1, gt=0)
    num_preprocess_cells: int = Field(default=3, gt=0)
    num_postprocess_cells: int = Field(default=2, gt=0)
    groups_per_scale: int = Field(default=2, gt=0)
    latent_per_group: int = Field(default=8, gt=0)
    score_channels: int = Field(default=64, gt=0)
    denoise_step: float = Field(default=0.001, ge=0.0)
    clamp: float = Field(default=1.0, gt=0.0)
    psi: float = Field(default=0.5, ge=0.0)
    lambda_dsm: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _beta_schedule_is_ordered(self) -> "ModelParameterConfig":
        if self.beta_start > self.beta_end:
            raise ValueError("need 0 <= beta_start <= beta_end < 1")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        **params,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Replace the configured loss by the paper objective (Eq. 14).

    The coupled diffusion needs the true target, and the pass runs on diffused
    inputs, so it produces no forecast.
    """
    return None, model.training_loss(batch.x, batch.x_mark, batch.target)


SPEC = ModelSpec(
    name="D3VAE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_d3vae.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "marks"),
    contract_task={"seq_len": 16, "pred_len": 16, "label_len": 0},
    training_objective=training_objective,
)
