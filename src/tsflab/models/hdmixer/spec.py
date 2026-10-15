"""Runtime specification for HDMixer."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.hdmixer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_ff: int = Field(default=256, gt=0)
    e_layers: int = Field(default=3, gt=0)
    patch_len: int = Field(default=16, gt=1)
    stride: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    head_dropout: float = Field(default=0.0, ge=0, lt=1)
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    deform_range: float = Field(default=0.25, gt=0, le=1)
    mix_time: bool = True
    mix_variable: bool = True
    mix_channel: bool = True


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="HDMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
