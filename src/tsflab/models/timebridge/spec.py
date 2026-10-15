"""Model specification for TimeBridge."""

from __future__ import annotations

from pydantic import BaseModel

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timebridge.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    num_p: int | None = None
    ia_layers: int = 2
    pd_layers: int = 1
    ca_layers: int = 2
    stable_len: int = 3
    d_model: int = 16
    n_heads: int = 4
    d_ff: int = 128
    attn_dropout: float = 0.15
    dropout: float = 0.0
    activation: str = "gelu"
    revin: bool = True


def build_model(cfg, params):
    """Construct TimeBridge from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        period=params["period"],
        num_p=params.get("num_p"),
        ia_layers=params["ia_layers"],
        pd_layers=params["pd_layers"],
        ca_layers=params["ca_layers"],
        stable_len=params["stable_len"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_ff=params["d_ff"],
        attn_dropout=params["attn_dropout"],
        dropout=params["dropout"],
        activation=params["activation"],
        revin=bool(params["revin"]),
    )


SPEC = ModelSpec(
    name="TimeBridge",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
