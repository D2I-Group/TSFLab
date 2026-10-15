"""Runtime specification for PGN."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pgn.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    period: int = Field(default=24, gt=0)
    d_model: int = Field(default=64, gt=0)
    norm: bool = True
    use_short_branch: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        period=params["period"],
        d_model=params["d_model"],
        norm=params["norm"],
        use_short_branch=params["use_short_branch"],
    )


SPEC = ModelSpec(
    name="PGN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_pgn.toml",
    capabilities=frozenset({"time-series"}),
    components=("marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
