"""Runtime specification for MambaSimple."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mambasimple.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    c_out: int | None = Field(default=None, gt=0)
    d_model: int = Field(default=128, gt=0)
    d_state: int = Field(default=16, gt=0)
    e_layers: int = Field(default=2, gt=0)
    expand: int = Field(default=2, gt=0)
    d_conv: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def channels_match(self):
        if self.c_out not in {None, self.enc_in}:
            raise ValueError("c_out must equal enc_in")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="MambaSimple",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("mamba",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
