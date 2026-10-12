"""Runtime specification for SWIFT."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.swift.model import Model

"""Validated parameters for SWIFT."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    hidden_size: int = 0
    conv_kernel: int = 3
    use_convdropout: bool = False
    conv_dropout: float = 0.1
    not_independent: bool = False


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        hidden_size=params["hidden_size"],
        conv_kernel=params["conv_kernel"],
        use_convdropout=params["use_convdropout"],
        conv_dropout=params["conv_dropout"],
        not_independent=params["not_independent"],
    )


SPEC = ModelSpec(
    name="SWIFT",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_swift.toml",
    capabilities=frozenset({"time-series"}),
    components=("haar_dwt1d", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
