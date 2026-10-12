"""Runtime specification for ARIMATS."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.arima_ts.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    ar_order: int = Field(default=2, ge=0)
    ma_order: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _one_coefficient(self):
        if self.ar_order + self.ma_order < 1:
            raise ValueError("at least one AR or MA coefficient is required")
        return self


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
