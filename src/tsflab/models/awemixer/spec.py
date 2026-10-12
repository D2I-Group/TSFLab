"""Runtime specification for AWEMixer."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.awemixer.model import Model

"""Validated parameters for AWEMixer."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    dropout: float = 0.2
    num_scales: int = 3
    wavelet_level: int = 3
    wavelet: str = "db4"
    num_fusion_layers: int = 1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        num_scales=params["num_scales"],
        wavelet_level=params["wavelet_level"],
        wavelet=params["wavelet"],
        num_fusion_layers=params["num_fusion_layers"],
    )


SPEC = ModelSpec(
    name="AWEMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_awemixer.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin", "wavelet"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
