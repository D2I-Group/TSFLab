"""Runtime specification for WDformer."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wdformer.model import Model

"""Validated parameters for WDformer."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    wave_size: int = 3


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        wave_size=params["wave_size"],
    )


SPEC = ModelSpec(
    name="WDformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_wdformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("differential_attention", "instance_norm", "wavelet"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
