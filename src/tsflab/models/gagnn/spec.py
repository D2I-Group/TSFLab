"""Model specification for GAGNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gagnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GAGNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    group_num: int = 4


def build_model(cfg, params):
    """Construct GAGNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        adj_mx=params.get("adj_mx"),
        cov_dim=params["cov_dim"],
        d_model=params["d_model"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        group_num=params["group_num"],
    )


SPEC = ModelSpec(
    name="GAGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
