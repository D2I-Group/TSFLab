"""Runtime specification for SVRForecasterTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.svr_forecaster_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_support: int = 16
    kernel_gamma: float = 0.1
    epsilon: float = 0.1
    l2_penalty: float = 1e-4


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["num_support"],
        params["kernel_gamma"],
        params["epsilon"],
        params["l2_penalty"],
    )


SPEC = ModelSpec(
    name="SVRForecasterTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
