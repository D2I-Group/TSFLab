"""Runtime specification for NSTransformer."""

from pydantic import BaseModel, Field, model_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.nstransformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(128, gt=0)
    n_heads: int = Field(8, gt=0)
    e_layers: int = Field(2, gt=0)
    d_layers: int = Field(1, gt=0)
    d_ff: int = Field(256, gt=0)
    dropout: float = Field(0.1, ge=0, lt=1)
    p_hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])
    p_hidden_layers: int = Field(2, gt=0)

    @model_validator(mode="after")
    def valid(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if len(self.p_hidden_dims) < self.p_hidden_layers or any(
            v <= 0 for v in self.p_hidden_dims
        ):
            raise ValueError("projector hidden dimensions are incomplete")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        cfg.task.label_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="NSTransformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
