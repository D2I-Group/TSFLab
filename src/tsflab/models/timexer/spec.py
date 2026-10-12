"""Model specification for TimeXer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timexer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    patch_len: int = 16
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True


def build_model(cfg, params):
    """Construct TimeXer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_ff=params["d_ff"],
        patch_len=params["patch_len"],
        dropout=params["dropout"],
        activation=params["activation"],
        use_norm=bool(params["use_norm"]),
    )


SPEC = ModelSpec(
    name="TimeXer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
