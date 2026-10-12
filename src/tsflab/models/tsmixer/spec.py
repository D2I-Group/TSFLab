"""Model specification for TSMixer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tsmixer.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


def build_model(cfg, params):
    """Construct TSMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        e_layers=params["e_layers"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="TSMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "mixer_block"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
