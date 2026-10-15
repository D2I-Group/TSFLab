"""Model specification for TiDE."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tide.model import Model


class ModelParameterConfig(BaseModel):
    d_model: int = Field(default=512, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    decoder_output_dim: int = Field(default=7, gt=0)
    time_feat_dim: int = Field(default=6, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    bias: bool = True
    feature_encode_dim: int = Field(default=2, gt=0)


def build_model(cfg, params):
    """Construct TiDE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        d_model=params["d_model"],
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        d_ff=params["d_ff"],
        decoder_output_dim=params["decoder_output_dim"],
        time_feat_dim=params["time_feat_dim"],
        dropout=params["dropout"],
        bias=bool(params["bias"]),
        feature_encode_dim=params["feature_encode_dim"],
    )


SPEC = ModelSpec(
    name="TiDE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("instance_norm",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
