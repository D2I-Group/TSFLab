"""Runtime specification for ElasticNetTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.elastic_net_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    penalty: float = 1e-4
    l1_ratio: float = 0.5


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
