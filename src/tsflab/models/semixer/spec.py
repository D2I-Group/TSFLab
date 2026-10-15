"""Model specification for SEMixer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.semixer.model import Model, _parse_scale_factors


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    scale_factors: str = "1x2x4x8"
    reduce_dim: int = Field(default=64, gt=0)
    eib_num: int = Field(default=1, gt=0)
    eib_num_1scale: int = Field(default=1, gt=0)
    connection_probability: float = Field(default=0.85, ge=0.0, lt=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    mixing_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    affine: bool = False
    subtract_last: bool = False

    @model_validator(mode="after")
    def _architecture_contract(self) -> ModelParameterConfig:
        if self.enc_in != self.c_out:
            raise ValueError("enc_in and c_out must match")
        # Raises ValueError on a malformed or non-increasing scale schedule.
        _parse_scale_factors(self.scale_factors)
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="SEMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_semixer.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "positional_encoding", "revin"),
    contract_task={"seq_len": 336, "pred_len": 96, "label_len": 0},
)
