"""Model specification for PaiFilter."""

from __future__ import annotations

from pydantic import BaseModel

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.paifilter.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 256


def build_model(cfg, params):
    """Construct PaiFilter from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        hidden_size=params["hidden_size"],
    )


SPEC = ModelSpec(
    name="PaiFilter",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
