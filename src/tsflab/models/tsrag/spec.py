"""Runtime specification for TSRAG."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tsrag.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    top_k: int = Field(default=4, ge=1)
    memory_size: int = Field(default=8, ge=1)
    num_heads: int = Field(default=4, ge=1)
    dropout: float = 0.1

    @model_validator(mode="after")
    def _architecture_contract(self):
        if self.top_k > self.memory_size:
            raise ValueError("top_k cannot exceed memory_size")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        top_k=params["top_k"],
        memory_size=params["memory_size"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="TSRAG",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
