"""Runtime specification for S4."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.s4.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    c_out: int | None = Field(default=None, gt=0)
    d_model: int = Field(default=128, gt=0)
    d_state: int = Field(default=64, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    use_norm: bool = True

    @model_validator(mode="after")
    def architecture(self):
        if self.d_state % 2:
            raise ValueError("d_state must be even")
        if self.use_norm and self.c_out not in {None, self.enc_in}:
            raise ValueError("normalized S4 requires c_out == enc_in")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="S4",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
