"""Runtime specification for MMPD."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.mmpd.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    dropout: float = 0.1
    patch_len: int = Field(default=8, ge=1)
    num_heads: int = Field(default=4, ge=1)
    adjacent_range: int = Field(default=1, ge=0)
    diffusion_steps: int = Field(default=100, ge=1)
    denoiser_depth: int = 2
    diffusion_weight: float = Field(default=0.99, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_heads(self) -> ModelParameterConfig:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        patch_len=params["patch_len"],
        num_heads=params["num_heads"],
        adjacent_range=params["adjacent_range"],
        diffusion_steps=params["diffusion_steps"],
        denoiser_depth=params["denoiser_depth"],
        diffusion_weight=params["diffusion_weight"],
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """MMPD Eq. (8): diffusion plus anchor loss replaces the observation loss."""
    return None, model.diffusion_loss(batch.x, batch.target)


SPEC = ModelSpec(
    name="MMPD",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
