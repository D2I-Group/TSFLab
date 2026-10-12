"""Model specification for CATS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cats.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 24
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    stride: int = 24
    attn_dropout: float = 0.0
    query_independence: bool = False
    padding_patch: str | None = None
    store_attn: bool = False
    QAM_start: float = 0.1
    QAM_end: float = 0.5


def build_model(cfg, params):
    """Construct CATS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        patch_len=params["patch_len"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_ff=params["d_ff"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        stride=params["stride"],
        attn_dropout=params["attn_dropout"],
        query_independence=params["query_independence"],
        padding_patch=params.get("padding_patch"),
        store_attn=params["store_attn"],
        QAM_start=params["QAM_start"],
        QAM_end=params["QAM_end"],
    )


SPEC = ModelSpec(
    name="CATS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
