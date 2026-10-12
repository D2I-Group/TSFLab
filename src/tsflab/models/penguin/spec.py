"""Runtime specification for PENGUIN."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.penguin.model import Model

"""Validated parameters for PENGUIN."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_ff: int = Field(default=256, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    dropout: float = 0.1
    periods: list[int] = [24]
    activation: Literal["relu", "gelu"] = "relu"
    use_rmsnorm: bool = False
    alibi: bool = True
    causal: bool = True

    @model_validator(mode="after")
    def validate_periods(self):
        if any(p < self.stride or p % self.stride for p in self.periods):
            raise ValueError("every period must be a positive multiple of stride")
        return self

    @model_validator(mode="after")
    def validate_heads(self):
        if self.d_model % self.n_heads or self.n_heads % (len(self.periods) or 1):
            raise ValueError("n_heads must divide d_model and be a multiple of the number of periods")
        return self


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
