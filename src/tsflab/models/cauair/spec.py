"""Runtime specification for CauAir."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cauair.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    dim: int = Field(default=64, gt=0)
    cache_count: int = Field(default=8, gt=0)
    heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_heads(self):
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        return self


def build_model(cfg, params):
    params.pop("adj_mx", None)
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="CauAir",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
