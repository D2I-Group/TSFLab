"""Runtime specification for HMformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.hmformer.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    num_scales: int = 3
    depth: int = 1
    num_heads: int = 4


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        num_scales=params["num_scales"],
        depth=params["depth"],
        num_heads=params["num_heads"],
    )


SPEC = ModelSpec(
    name="HMformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
