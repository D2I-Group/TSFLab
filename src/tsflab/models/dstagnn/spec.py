"""Model specification for DSTAGNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dstagnn.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated DSTAGNN parameters supplied via ``model.params``."""

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    d_k: int = Field(default=8, ge=1)
    d_v: int = Field(default=8, ge=1)
    n_heads: int = Field(default=4, ge=1)


def build_model(cfg, params):
    """Construct DSTAGNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        adj_mx=params.get("adj_mx"),
        d_model=params["d_model"],
        d_k=params["d_k"],
        d_v=params["d_v"],
        n_heads=params["n_heads"],
    )


SPEC = ModelSpec(
    name="DSTAGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_spectral",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
