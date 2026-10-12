"""Runtime specification for ReCycle."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.recycle.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``task.seq_len`` and ``task.pred_len`` must be whole multiples of ``cycle_len``
    (the paper: 21 and 7 days of 24 hourly steps).
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    cycle_len: int = Field(default=24, gt=0)
    rhp_cycles: int = Field(default=3, gt=0)
    rhp_mode: Literal["last", "causal"] = "last"
    d_model: int = Field(default=32, gt=0)
    n_heads: int = Field(default=1, gt=0)
    e_layers: int = Field(default=1, gt=0)
    d_layers: int = Field(default=1, gt=0)
    d_ff: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="ReCycle",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_recycle.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 504, "pred_len": 168, "label_len": 0},
)
