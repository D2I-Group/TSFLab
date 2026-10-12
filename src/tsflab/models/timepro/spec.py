"""Runtime specification for TimePro."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timepro.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    d_model: int = Field(default=16, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_state: int = Field(default=1, gt=0)
    d_conv: int = Field(default=5, gt=0)
    expand: float = Field(default=1.0, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_norm: bool = True

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_state != 1:
            raise ValueError("this local implementation supports only d_state=1")
        return self


def build_model(cfg, params):
    """Construct TimePro from a validated run configuration."""
    patch_len = params["patch_len"]
    stride = params["stride"]
    d_model = params["d_model"]
    expand = params["expand"]
    if patch_len > cfg.task.seq_len:
        raise ValueError("patch_len must not exceed seq_len")
    patch_num = (cfg.task.seq_len + stride - patch_len) // stride + 1
    d_inner = int(expand * d_model * patch_num)
    if d_inner % patch_num:
        raise ValueError("expand * d_model * patch_num must be an integer multiple of patch_num")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        e_layers=params["e_layers"],
        d_state=params["d_state"],
        d_conv=params["d_conv"],
        expand=expand,
        dropout=params["dropout"],
        use_norm=bool(params["use_norm"]),
    )


SPEC = ModelSpec(
    name="TimePro",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_timepro.toml",
    capabilities=frozenset({"time-series"}),
    components=("hyper_state_scan",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
