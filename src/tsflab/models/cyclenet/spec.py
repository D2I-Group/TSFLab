"""Model specification for CycleNet."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cyclenet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    cycle: int = 24
    model_type: str = "linear"
    d_model: int = 512
    use_revin: bool = True


def build_model(cfg, params):
    """Construct CycleNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cycle=params.get('cycle', 24), model_type=params.get('model_type', 'linear'), d_model=params.get('d_model', 512), use_revin=bool(params.get('use_revin', True)))
    )


SPEC = ModelSpec(
    name='CycleNet',
    module='tsflab.models.cyclenet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/CycleNet.toml',
    model_card='src/tsflab/models/cyclenet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('channel_wise_linear', 'marks', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
