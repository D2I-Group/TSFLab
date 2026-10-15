"""Runtime specification for GaussianProcessTS."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gaussian_process_ts.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_inducing: int = Field(default=16, gt=0)
    length_scale: float = Field(default=1.0, gt=0)
    noise: float = Field(default=1e-3, gt=0)


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
