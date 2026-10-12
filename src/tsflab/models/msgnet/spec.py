"""Runtime specification for MSGNet."""

from pydantic import BaseModel, Field, model_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.msgnet.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    e_layers: int = Field(default=2, gt=0)
    n_heads: int = Field(default=8, gt=0)
    top_k: int = Field(default=5, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    gcn_depth: int = Field(default=2, gt=0)
    propalpha: float = Field(default=0.3, ge=0, le=1)
    node_dim: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def architecture(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="MSGNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("dominant_periods", "instance_norm"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
