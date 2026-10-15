"""Runtime specification for SymTime."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.symtime.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_len: int = Field(default=16, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=4, gt=0)
    trend_kernel: int = Field(default=25, gt=0)
    dropout: float = 0.1

    @model_validator(mode="after")
    def _architecture(self):
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.trend_kernel % 2 == 0:
            raise ValueError("trend_kernel must be a positive odd integer")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        trend_kernel=params["trend_kernel"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="SymTime",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
