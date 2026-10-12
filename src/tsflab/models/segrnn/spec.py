"""Model specification for SegRNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.segrnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    seg_len: int = 24


def build_model(cfg, params):
    """Construct SegRNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        seg_len=params["seg_len"],
    )


SPEC = ModelSpec(
    name="SegRNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("last_value_center",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
