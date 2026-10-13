"""Construction and runtime contract for AdaRDiff."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.adardiff.model import Model


class ModelParameterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enc_in: int = Field(default=7, gt=0)
    period: int = Field(default=24, gt=0)
    enabled: bool = True
    phase1_epochs: int = Field(default=30, ge=0)
    phase1_lr: float = Field(default=0.001, gt=0)
    phase1_patience: int = Field(default=3, gt=0)


def build_model(cfg, params):
    if cfg.task.seq_len < max(params["period"], 2):
        raise ValueError("AdaRDiff needs seq_len >= max(period, 2)")
    return Model(
        channels=params["enc_in"],
        lookback=cfg.task.seq_len,
        horizon=cfg.task.pred_len,
        p=params["period"],
        enabled=params["enabled"],
        phase1_epochs=params["phase1_epochs"],
        phase1_lr=params["phase1_lr"],
        phase1_patience=params["phase1_patience"],
    )


SPEC = ModelSpec(
    name="AdaRDiff",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series", "pretraining-stage"]),
    components=("channel_wise_linear",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
