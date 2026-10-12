"""Runtime specification for MLPForecasterTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mlp_forecaster_ts.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    dropout: float = 0.1
    num_layers: int = 1
    use_revin: bool = True


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="MLPForecasterTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
