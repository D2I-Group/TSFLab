"""Model specification for PM25_GNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pm25gnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PM25_GNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64


def build_model(cfg, params):
    """Construct PM25_GNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        adj_mx=params.get("adj_mx"),
        cov_dim=params["cov_dim"],
        hid_dim=params["hid_dim"],
    )


SPEC = ModelSpec(
    name="PM25_GNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("graph_conv_gru", "marks"),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
