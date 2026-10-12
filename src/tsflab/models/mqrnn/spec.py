"""Model specification for MQRNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mqrnn.model import Model, check_quantile_levels

from pydantic import BaseModel, Field, field_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    context_size: int = Field(default=32, gt=0)
    decoder_hidden: int = Field(default=64, gt=0)
    future_covariate_size: int = Field(default=6, ge=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    quantile_levels: list[float] | None = None

    @field_validator("quantile_levels")
    @classmethod
    def _quantile_levels(cls, values: list[float] | None) -> list[float] | None:
        if values is not None:
            check_quantile_levels(values)
        return values


def build_model(cfg, params):
    """Construct MQRNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        features=cfg.task.features,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        context_size=params["context_size"],
        decoder_hidden=params["decoder_hidden"],
        future_covariate_size=params["future_covariate_size"],
        dropout=params["dropout"],
        quantile_levels=params.get("quantile_levels") or list(cfg.evaluation.quantile_levels),
    )


SPEC = ModelSpec(
    name="MQRNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate", "quantile-output", "time-series"]),
    components=("quantile_head",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
