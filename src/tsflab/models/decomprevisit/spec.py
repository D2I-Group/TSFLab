"""Runtime specification for DecompRevisit."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.decomprevisit.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    kernel_size: int = Field(default=25, gt=0)
    trend_hidden: int = Field(default=512, gt=0)
    seasonal_head: Literal["shift_mlp", "mlp"] = "shift_mlp"
    shift_hidden: int = Field(default=64, gt=0)
    shift_mix_hidden: int = Field(default=128, gt=0)

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="DecompRevisit",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_decomprevisit.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
