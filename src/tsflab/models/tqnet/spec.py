"""Runtime specification for TQNet."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tqnet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    cycle: int = Field(default=24, gt=0)
    d_model: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    attn_dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    channel_aggre_heads: int = Field(default=4, gt=0)
    use_revin: bool = True

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        return self


def build_model(cfg, params):
    """Construct TQNet from a validated run configuration."""
    if cfg.task.seq_len % params["channel_aggre_heads"]:
        raise ValueError("seq_len must be divisible by channel_aggre_heads")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        cycle=params["cycle"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        attn_dropout=params["attn_dropout"],
        channel_aggre_heads=params["channel_aggre_heads"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="TQNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_tqnet.toml",
    capabilities=frozenset({"time-series"}),
    components=("periodic_query_bank", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
