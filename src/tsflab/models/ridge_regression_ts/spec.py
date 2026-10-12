"""Runtime specification for RidgeRegressionTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.ridge_regression_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    l2_penalty: float = 1e-4


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        l2_penalty=params["l2_penalty"],
    )


SPEC = ModelSpec(
    name="RidgeRegressionTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
