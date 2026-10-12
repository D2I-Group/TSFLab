"""Runtime specification for PULSE."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pulse.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    phase_period: int = 24
    phase_resolution: int = 8
    router_heads: int = 4
    dropout: float = 0.1
    eps: float = 1e-5


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        phase_period=params["phase_period"],
        phase_resolution=params["phase_resolution"],
        router_heads=params["router_heads"],
        dropout=params["dropout"],
        eps=params["eps"],
    )


SPEC = ModelSpec(
    name="PULSE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
