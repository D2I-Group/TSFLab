"""Runtime specification for TSLIF."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tslif.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    hidden_size: int = Field(default=128, gt=0)
    layers: int = Field(default=1, gt=0)
    kernel_size: int = Field(default=3, gt=0)
    v_threshold: float = Field(default=1.0, gt=0.0)
    surrogate_alpha: float = Field(default=2.0, gt=0.0)


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="TSLIF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_tslif.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
