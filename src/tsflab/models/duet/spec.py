"""Runtime specification for DUET."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.duet.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(64, gt=0)
    n_heads: int = Field(4, gt=0)
    e_layers: int = Field(2, gt=0)
    d_ff: int = Field(64, gt=0)
    dropout: float = Field(0.1, ge=0, lt=1)
    fc_dropout: float = Field(0.1, ge=0, lt=1)
    moving_avg: int = Field(25, gt=0)
    num_experts: int = Field(4, gt=0)
    k: int = Field(2, gt=0)
    hidden_size: int = Field(64, gt=0)
    noisy_gating: bool = True

    @model_validator(mode="after")
    def valid(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.k > self.num_experts:
            raise ValueError("k cannot exceed num_experts")
        return self


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, features=cfg.task.features, **params)


SPEC = ModelSpec(
    name="DUET",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "topk_expert_router"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
