"""Runtime specification for GTR."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gtr.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    cycle_length: int = 168
    local_period: int = 24
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params.get("d_model", 64),
        dropout=params.get("dropout", 0.1),
        cycle_length=params.get("cycle_length", 168),
        local_period=params.get("local_period", 24),
        use_revin=bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name="GTR",
    module="tsflab.models.gtr",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/GTR.toml",
    model_card="src/tsflab/models/gtr/README.md",
    smoke_config=None,
    capabilities=frozenset(["time-series"]),
    components=("marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
