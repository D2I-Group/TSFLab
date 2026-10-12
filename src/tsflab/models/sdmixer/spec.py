"""Runtime specification for SDMixer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sdmixer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    spectral_top_k: int = Field(default=5, gt=0)
    channel_sparse_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    attn_sparse_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    d_ff: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.enc_in != self.c_out:
            raise ValueError("enc_in and c_out must match (channel-preserving forecaster)")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        c_out=params["c_out"],
        spectral_top_k=params["spectral_top_k"],
        channel_sparse_ratio=params["channel_sparse_ratio"],
        attn_sparse_ratio=params["attn_sparse_ratio"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="SDMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_sdmixer.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
