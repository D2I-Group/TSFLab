"""Model specification for GaussianMLP."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gaussian_mlp.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    hidden_size: int = Field(default=256, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    eps: float = Field(default=1e-6, gt=0.0)


def build_model(cfg, params):
    """Construct GaussianMLP from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        features=cfg.task.features,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        eps=params["eps"],
    )


SPEC = ModelSpec(
    name="GaussianMLP",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gaussian_mlp.toml",
    capabilities=frozenset(["distribution-output", "time-series"]),
    components=("gaussian_parameter_head",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
