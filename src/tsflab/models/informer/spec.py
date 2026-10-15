"""Model specification for Informer."""

from __future__ import annotations

from pydantic import BaseModel

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.informer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int | None = None
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
    distil: bool = True
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct Informer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        dec_in=params.get("dec_in"),
        c_out=params.get("c_out"),
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        factor=params["factor"],
        activation=params["activation"],
        distil=bool(params["distil"]),
        embed=params["embed"],
        freq=params["freq"],
    )


SPEC = ModelSpec(
    name="Informer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("embed", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
