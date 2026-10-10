"""Model specification for TimeMosaic."""
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timemosaic.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_sizes: tuple[int, ...] = (4, 8, 16)
    num_segments: int = Field(default=4, gt=0)
    num_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    use_revin: bool = True
    selection_tau: float = Field(default=0.5, gt=0)
    budget_weight: float = Field(default=0.001, ge=0)


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params['enc_in'],
                 params.get('d_model', 64), tuple(params.get('patch_sizes', (4, 8, 16))),
                 params.get('num_segments', 4), params.get('num_heads', 4),
                 params.get('dropout', 0.1), bool(params.get('use_revin', True)),
                 float(params.get('selection_tau', 0.5)), float(params.get('budget_weight', 0.001)))


SPEC = ModelSpec(name='TimeMosaic', module='tsflab.models.timemosaic', model_class=Model,
                 factory=build_model, params_schema=ModelParameterConfig,
                 config_path='configs/models/TimeMosaic.toml', model_card='src/tsflab/models/timemosaic/README.md',
                 smoke_config=None, capabilities=frozenset(['time-series']),                  components=('revin',), contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0})
