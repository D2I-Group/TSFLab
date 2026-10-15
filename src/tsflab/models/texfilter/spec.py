"""Model specification for TexFilter."""

from __future__ import annotations

from pydantic import BaseModel

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.texfilter.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    embed_size: int = 128
    hidden_size: int = 256
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct TexFilter from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        embed_size=params["embed_size"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="TexFilter",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
