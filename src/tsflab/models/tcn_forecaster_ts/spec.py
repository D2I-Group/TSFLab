"""Runtime specification for TCNForecasterTS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tcn_forecaster_ts.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    num_layers: int = Field(default=2, ge=1)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(cfg.task.seq_len, cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="TCNForecasterTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
