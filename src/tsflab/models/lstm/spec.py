"""Model specification for LSTM."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.lstm.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    init_dim: int = Field(default=32, ge=1)
    hid_dim: int = Field(default=64, ge=1)
    end_dim: int = Field(default=128, ge=1)
    layer: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    cov_dim: int = Field(default=2, ge=0)


def build_model(cfg, params):
    """Construct LSTM from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        init_dim=params["init_dim"],
        hid_dim=params["hid_dim"],
        end_dim=params["end_dim"],
        layer=params["layer"],
        dropout=params["dropout"],
        cov_dim=params["cov_dim"],
    )


SPEC = ModelSpec(
    name="LSTM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
