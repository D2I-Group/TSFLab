"""Runtime specification for FreDF."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.fredf.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    e_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    freq: Literal["h"] = "h"


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        e_layers=params["e_layers"],
        dropout=params["dropout"],
        freq=params["freq"],
    )


SPEC = ModelSpec(
    name="FreDF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_fredf.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
