"""Runtime specification for Sensorformer."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sensorformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    n_heads: int = Field(default=4, ge=1)
    d_ff: int = Field(default=128, ge=1)
    layers: int = Field(default=2, ge=1)
    patch_len: int = Field(default=16, ge=1)
    stride: int = Field(default=8, ge=1)
    dropout: float = 0.1


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_ff=params["d_ff"],
        layers=params["layers"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="Sensorformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_sensorformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "flatten_forecast_head", "global_patch_compression_attention"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
