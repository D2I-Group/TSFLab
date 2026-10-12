"""Runtime specification for MultiPatchFormer."""

from pydantic import BaseModel, Field, model_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.multipatchformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(64, gt=0)
    n_heads: int = Field(4, gt=0)
    e_layers: int = Field(2, gt=0)
    d_ff: int = Field(128, gt=0)
    dropout: float = Field(0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def valid(self):
        if self.d_model % 4 or self.d_model % self.n_heads:
            raise ValueError("d_model must divide four scales and n_heads")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="MultiPatchFormer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
