"""Runtime specification for ARIMATS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.arima_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    ar_order: int = 2
    ma_order: int = 1


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["ar_order"],
        params["ma_order"],
    )


SPEC = ModelSpec(
    name="ARIMATS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
