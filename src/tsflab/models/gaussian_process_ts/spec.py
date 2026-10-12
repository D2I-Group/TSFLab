"""Runtime specification for GaussianProcessTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gaussian_process_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_inducing: int = 16
    length_scale: float = 1.0
    noise: float = 1e-3


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["num_inducing"],
        params["length_scale"],
        params["noise"],
    )


SPEC = ModelSpec(
    name="GaussianProcessTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
