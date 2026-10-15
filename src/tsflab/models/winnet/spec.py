"""Runtime specification for WinNet."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.winnet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    window: int = Field(default=24, gt=1)
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="WinNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_winnet.toml",
    capabilities=frozenset({"time-series"}),
    components=("last_value_center",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
