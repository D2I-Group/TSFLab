"""Model specification for MGSFformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mgsfformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MGSFformer parameters supplied via ``model.params``."""

    enc_in: int
    IE_dim: int = 32
    dropout: float = 0.3
    num_head: int = 2


def build_model(cfg, params):
    """Construct MGSFformer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        IE_dim=params["IE_dim"],
        dropout=params["dropout"],
        num_head=params["num_head"],
    )


SPEC = ModelSpec(
    name="MGSFformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("revin",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
