"""Runtime specification for MMPD."""

from __future__ import annotations

from tsflab.experiments.runner.objective import TrainingBatch

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mmpd.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    patch_len: int = 8
    num_heads: int = 4
    adjacent_range: int = 1
    diffusion_steps: int = 100
    denoiser_depth: int = 2
    diffusion_weight: float = 0.99


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
