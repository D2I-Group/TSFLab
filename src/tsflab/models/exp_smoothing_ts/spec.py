"""Runtime specification for ExpSmoothingTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.exp_smoothing_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    initial_alpha: float = 0.5


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        initial_alpha=params["initial_alpha"],
    )


SPEC = ModelSpec(
    name="ExpSmoothingTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
