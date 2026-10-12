"""Runtime specification for PolynomialRegressionTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.polynomial_regression_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    degree: int = 2


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        degree=params["degree"],
    )


SPEC = ModelSpec(
    name="PolynomialRegressionTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
