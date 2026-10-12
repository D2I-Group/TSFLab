"""Runtime specification for PatchTSMixer."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.patchtsmixer.model import Model

"""Validated parameters for PatchTSMixer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 16
    patch_len: int = 16
    stride: int = 8
    num_layers: int = 3
    expansion_factor: int = 2
    dropout: float = 0.1
    head_dropout: float = 0.1
    mode: Literal["common_channel", "mix_channel"] = "common_channel"
    gated_attn: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        num_layers=params["num_layers"],
        expansion_factor=params["expansion_factor"],
        dropout=params["dropout"],
        head_dropout=params["head_dropout"],
        mode=params["mode"],
        gated_attn=params["gated_attn"],
    )


SPEC = ModelSpec(
    name="PatchTSMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_patchtsmixer.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "revin", "softmax_gate"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
