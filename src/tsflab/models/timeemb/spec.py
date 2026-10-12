"""Model specification for TimeEmb."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timeemb.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 512
    use_revin: bool = True
    use_hour_index: bool = True
    use_day_index: bool = False
    scale: float = 0.02
    hour_length: int = 24
    day_length: int = 7


def build_model(cfg, params):
    """Construct TimeEmb from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        use_revin=bool(params["use_revin"]),
        use_hour_index=bool(params["use_hour_index"]),
        use_day_index=bool(params["use_day_index"]),
        scale=params["scale"],
        hour_length=params["hour_length"],
        day_length=params["day_length"],
    )


SPEC = ModelSpec(
    name="TimeEmb",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
