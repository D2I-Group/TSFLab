"""Runtime specification for SCINet."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.scinet.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_stacks: int = Field(default=1, ge=1, le=2)
    num_levels: int = Field(default=3, gt=0)
    hidden_size: int | None = Field(default=None, gt=0)
    kernel_size: int = Field(default=5, gt=0)
    positional_encoding: bool = False
    instance_norm: bool = False
    dropout: float = Field(default=0.0, ge=0, lt=1)


def build_model(cfg, params):
    block = 2 ** params["num_levels"]
    seq_len, pred_len = cfg.task.seq_len, cfg.task.pred_len
    if seq_len % block or (params["num_stacks"] == 2 and (seq_len + pred_len) % block):
        raise ValueError("seq_len (and seq_len + pred_len with two stacks) must be divisible by 2**num_levels")
    return Model(seq_len=seq_len, pred_len=pred_len, **params)


SPEC = ModelSpec(
    name="SCINet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("instance_norm",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
