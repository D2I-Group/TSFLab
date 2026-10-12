"""Model specification for STTN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sttn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STTN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    mlp_expand: int = 4
    num_layers: int = 3
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct STTN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        adj_mx=params.get("adj_mx"),
        cov_dim=params["cov_dim"],
        d_model=params["d_model"],
        mlp_expand=params["mlp_expand"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="STTN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
