"""Model specification for InterPDN."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.interpdn.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    support_size: int = Field(default=31, ge=3)
    support_bound: float = Field(default=4.0, gt=0)
    ema_decay: float = Field(default=0.8, gt=0, lt=1)
    use_revin: bool = True


def build_model(cfg, params):
    """Construct InterPDN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        support_size=params["support_size"],
        support_bound=params["support_bound"],
        ema_decay=params["ema_decay"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="InterPDN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
