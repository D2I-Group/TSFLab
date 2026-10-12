"""Runtime specification for DTAF."""

from pydantic import BaseModel, Field, model_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dtaf.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=32, gt=0)
    e_layers: int = Field(default=1, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    heads: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    expert_num: int = Field(default=2, gt=0)
    expert_hidden: int = Field(default=8, gt=0)
    top_k: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def architecture(self):
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="DTAF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("instance_norm",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
