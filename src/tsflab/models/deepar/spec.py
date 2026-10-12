"""Model specification for DeepAR."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.deepar.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    embedding_size: int = Field(default=32, ge=1)
    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    cov_feat_size: int = Field(default=0, ge=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    checkpoint_steps: int = Field(default=24, ge=0)


def build_model(cfg, params):
    """Construct DeepAR from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        embedding_size=params["embedding_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        cov_feat_size=params["cov_feat_size"],
        dropout=params["dropout"],
        checkpoint_steps=params["checkpoint_steps"],
    )


SPEC = ModelSpec(
    name="DeepAR",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["distribution-output", "time-series"]),
    components=("gaussian_parameter_head",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
