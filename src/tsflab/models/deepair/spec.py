"""Runtime specification for DeepAir."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.deepair.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    hidden_dim: int = Field(default=32, gt=0)
    spatial_regions: int = Field(default=4, gt=0)


def build_model(cfg, params):
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len, spatial_mx=params.pop("adj_mx", None), **params)


SPEC = ModelSpec(
    name="DeepAir",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
