"""Runtime specification for KNNForecasterTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.knn_forecaster_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_prototypes: int = 32
    kernel_gamma: float = 0.08


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
