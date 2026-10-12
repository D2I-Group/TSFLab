"""Model specification for PhaseFormer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.phaseformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 16
    dropout: float = 0.0
    period: int = 24
    num_routers: int = 4
    num_layers: int = 1
    num_heads: int = 1
    use_revin: bool = True


def build_model(cfg, params):
    """Construct PhaseFormer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        period=params["period"],
        num_routers=params["num_routers"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="PhaseFormer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
