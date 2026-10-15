"""Runtime specification for SVRForecasterTS."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.svr_forecaster_ts.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_support: int = Field(default=16, gt=0)
    kernel_gamma: float = Field(default=0.1, gt=0)
    epsilon: float = Field(default=0.1, ge=0)
    l2_penalty: float = Field(default=1e-4, ge=0)


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
