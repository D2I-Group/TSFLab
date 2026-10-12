"""Runtime specification for AutoRegressiveTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.autoregressive_ts.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"])


SPEC = ModelSpec(
    name="AutoRegressiveTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
