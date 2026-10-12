"""Runtime specification for TimeMachine."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timemachine.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    n1: int = Field(default=256, ge=1)
    n2: int = Field(default=128, ge=1)
    d_state: int = Field(default=256, ge=1)
    d_conv: int = Field(default=2, ge=1)
    expand: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    revin: bool = True
    ch_ind: bool = True
    residual: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        n1=params["n1"],
        n2=params["n2"],
        d_state=params["d_state"],
        d_conv=params["d_conv"],
        expand=params["expand"],
        dropout=params["dropout"],
        revin=params["revin"],
        ch_ind=params["ch_ind"],
        residual=params["residual"],
    )


SPEC = ModelSpec(
    name="TimeMachine",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_timemachine.toml",
    capabilities=frozenset({"time-series"}),
    components=("mamba", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
