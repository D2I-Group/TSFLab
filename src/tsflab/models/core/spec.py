"""Runtime specification for CoRe."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.core.model import Model

"""Validated parameters for CoRe."""

from pydantic import BaseModel, ConfigDict, Field


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    context_len: int = Field(default=5, gt=0)
    var_wise_gating: bool = True
    scr_rank: int = Field(default=0, ge=0)  # 0 selects rank C
    gate_init: float = 0.0
    gate_bias_init: float = -1.0


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        context_len=params["context_len"],
        var_wise_gating=params["var_wise_gating"],
        scr_rank=params["scr_rank"],
        gate_init=params["gate_init"],
        gate_bias_init=params["gate_bias_init"],
    )


SPEC = ModelSpec(
    name="CoRe",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_core.toml",
    capabilities=frozenset({"time-series", "test-time-adaptation"}),
    components=("channel_wise_linear", "spectral_descriptor"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
