"""Runtime specification for Reformer."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.reformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    c_out: int | None = Field(default=None, gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    bucket_size: int = Field(default=4, gt=0)
    n_hashes: int = Field(default=4, gt=0)
    causal: bool = False

    @model_validator(mode="after")
    def dimensions(self):
        if self.d_model % (2 * self.n_heads):
            raise ValueError("d_model/2 must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="Reformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
