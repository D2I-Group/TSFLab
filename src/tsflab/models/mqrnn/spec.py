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
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, hidden_size=params.get('hidden_size', 64), num_layers=params.get('num_layers', 1), context_size=params.get('context_size', 32), decoder_hidden=params.get('decoder_hidden', 64), future_covariate_size=params.get('future_covariate_size', 6), dropout=params.get('dropout', 0.1), quantile_levels=params.get('quantile_levels') or list(cfg.evaluation.quantile_levels))
    )


SPEC = ModelSpec(
    name='MQRNN',
    module='tsflab.models.mqrnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MQRNN.toml',
    model_card='src/tsflab/models/mqrnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate', 'quantile-output', 'time-series']),
    components=('quantile_head',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
