"""Runtime specification for ReFocus."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.refocus.model import Model

"""Validated parameters for ReFocus."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    d_pick: int = 32
    layers: int = 2
    dropout: float = 0.1
    beta: float = 0.5
    kernel_size: int = 25
    initial: bool = True


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        d_pick=params["d_pick"],
        layers=params["layers"],
        dropout=params["dropout"],
        beta=params["beta"],
        kernel_size=params["kernel_size"],
        initial=params["initial"],
    )


SPEC = ModelSpec(
    name="ReFocus",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_refocus.toml",
    capabilities=frozenset({"time-series"}),
    components=("energy_frequency_pooling", "revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
