"""Runtime specification for AWEMixer."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.awemixer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=128, ge=1)
    dropout: float = 0.2
    num_scales: int = Field(default=3, ge=1)
    wavelet_level: int = Field(default=3, ge=1)
    wavelet: str = "db4"
    num_fusion_layers: int = Field(default=1, ge=1)


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
