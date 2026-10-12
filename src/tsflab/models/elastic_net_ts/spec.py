"""Runtime specification for ElasticNetTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.elastic_net_ts.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    penalty: float = Field(default=1e-4, ge=0)
    l1_ratio: float = Field(default=0.5, ge=0, le=1)


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params["penalty"], params["l1_ratio"]
    )


SPEC = ModelSpec(
    name="ElasticNetTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
