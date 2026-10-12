"""Runtime specification for PENGUIN."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.penguin.model import Model

"""Validated parameters for PENGUIN."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 8
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    periods: list[int] = [24]
    activation: Literal["relu", "gelu"] = "relu"
    use_rmsnorm: bool = False
    alibi: bool = True
    causal: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        dropout=params["dropout"],
        periods=params["periods"],
        activation=params["activation"],
        use_rmsnorm=params["use_rmsnorm"],
        alibi=params["alibi"],
        causal=params["causal"],
    )


SPEC = ModelSpec(
    name="PENGUIN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_penguin.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "flatten_forecast_head", "mamba", "periodic_alibi_bias", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
