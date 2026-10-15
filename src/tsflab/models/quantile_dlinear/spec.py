"""Model specification for QuantileDLinear."""

from __future__ import annotations

from pydantic import BaseModel

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.quantile_dlinear.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    kernel_size: int = 25
    individual: bool = False
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    """Construct QuantileDLinear from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        features=cfg.task.features,
        kernel_size=params["kernel_size"],
        individual=bool(params["individual"]),
        quantile_levels=params.get("quantile_levels") or list(cfg.evaluation.quantile_levels),
    )


SPEC = ModelSpec(
    name="QuantileDLinear",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_quantile_dlinear.toml",
    capabilities=frozenset(["quantile-output", "time-series"]),
    components=("dlinear", "quantile_head"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
