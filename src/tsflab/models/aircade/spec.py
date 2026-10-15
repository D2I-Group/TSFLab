"""Runtime specification for AirCade."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.aircade.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    d_model: int = Field(default=32, gt=0)
    prompt_dim: int = Field(default=8, gt=0)
    adaptive_dim: int = Field(default=8, gt=0)
    num_heads: int = Field(default=4, gt=0)
    temporal_layers: int = Field(default=2, gt=0)
    spatial_layers: int = Field(default=2, gt=0)
    environments: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def validate_width(self):
        if self.d_model % self.num_heads or self.d_model <= 2 * self.prompt_dim:
            raise ValueError("d_model must divide across heads and exceed 2*prompt_dim")
        return self


def build_model(cfg, params):
    params.pop("adj_mx", None)
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="AirCade",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
)
