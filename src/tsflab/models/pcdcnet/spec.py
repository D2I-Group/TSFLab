"""Model specification for PCDCNet."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pcdcnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PCDCNet parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    cov_dim: int | None = None


def build_model(cfg, params):
    """Construct PCDCNet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        adj_mx=params.get("adj_mx"),
        cov_dim=params.get("cov_dim"),
        d_model=params["d_model"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="PCDCNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
