"""Model specification for UMixer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.umixer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    e_layers: int = Field(default=2, ge=1)
    patch_len: int = Field(default=16, ge=1)
    stride: int = Field(default=8, ge=1)
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct UMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        e_layers=params["e_layers"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="UMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
