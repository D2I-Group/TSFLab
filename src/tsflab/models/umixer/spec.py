"""Model specification for UMixer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.umixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
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
