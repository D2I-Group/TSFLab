"""Model specification for WPMixer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wpmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    dropout: float = 0.1
    tfactor: int = 5
    dfactor: int = 5
    wavelet: str = "db2"
    level: int = 1
    patch_len: int = 16
    stride: int = 8
    no_decomposition: bool = False


def build_model(cfg, params):
    """Construct WPMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        tfactor=params["tfactor"],
        dfactor=params["dfactor"],
        wavelet=params["wavelet"],
        level=params["level"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        no_decomposition=bool(params["no_decomposition"]),
    )


SPEC = ModelSpec(
    name="WPMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
