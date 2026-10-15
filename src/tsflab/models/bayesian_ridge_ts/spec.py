"""Runtime specification for BayesianRidgeTS."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.bayesian_ridge_ts.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    initial_weight_precision: float = Field(default=1e-3, gt=0)


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, params["enc_in"], params["initial_weight_precision"])


SPEC = ModelSpec(
    name="BayesianRidgeTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
