"""Model specification for BiST."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.bist.model import Model


class ModelParameterConfig(BaseModel):
    """Validated BiST parameters supplied via ``model.params``."""

    enc_in: int = Field(ge=1)
    model_dim: int = Field(default=32, ge=1)
    prompt_dim: int = Field(default=16, ge=1)
    num_layers: int = Field(default=3, ge=1)
    tod_size: int = Field(default=24, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    residual_steps: int = Field(default=2, ge=0)
    graph_dim: int = Field(default=8, ge=1)
    virtual_clusters: int = Field(default=8, ge=1)


def build_model(cfg, params):
    """Construct BiST from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        model_dim=params["model_dim"],
        prompt_dim=params["prompt_dim"],
        num_layers=params["num_layers"],
        tod_size=params["tod_size"],
        kernel_size=params["kernel_size"],
        residual_steps=params["residual_steps"],
        graph_dim=params["graph_dim"],
        virtual_clusters=params["virtual_clusters"],
    )


SPEC = ModelSpec(
    name="BiST",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks", "series_decomposition"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
