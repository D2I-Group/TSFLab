"""Construction and runtime contract for RhyMix."""
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.rhymix.model import Model


class ModelParameterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enc_in: int = Field(default=7, gt=0)
    dropout: float = Field(default=.1, ge=0, lt=1)


def build_model(cfg, params):
    return Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len,
                 pred_len=cfg.task.pred_len, dropout=params['dropout'])


SPEC = ModelSpec(
    name="RhyMix", module="tsflab.models.rhymix", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/RhyMix.toml",
    model_card="src/tsflab/models/rhymix/README.md", smoke_config=None,
    capabilities=frozenset(["time-series"]), components=(),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0},
)
