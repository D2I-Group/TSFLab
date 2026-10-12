"""Model specification for WaveNet."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wavenet.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    residual_channels: int = Field(default=16, ge=1)
    dilation_channels: int = Field(default=16, ge=1)
    skip_channels: int = Field(default=64, ge=1)
    end_channels: int = Field(default=128, ge=1)
    kernel_size: int = Field(default=2, ge=1)
    blocks: int = Field(default=2, ge=1)
    layers: int = Field(default=2, ge=1)
    use_norm: bool = True


def build_model(cfg, params):
    """Construct WaveNet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        residual_channels=params["residual_channels"],
        dilation_channels=params["dilation_channels"],
        skip_channels=params["skip_channels"],
        end_channels=params["end_channels"],
        kernel_size=params["kernel_size"],
        blocks=params["blocks"],
        layers=params["layers"],
        use_norm=bool(params["use_norm"]),
    )


SPEC = ModelSpec(
    name="WaveNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("gated_dilated_conv", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
