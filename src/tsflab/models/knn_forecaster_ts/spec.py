"""Runtime specification for KNNForecasterTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.knn_forecaster_ts.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_prototypes: int = Field(default=32, gt=0)
    kernel_gamma: float = Field(default=0.08, gt=0)


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        num_prototypes=params["num_prototypes"],
        kernel_gamma=params["kernel_gamma"],
    )


SPEC = ModelSpec(
    name="KNNForecasterTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
