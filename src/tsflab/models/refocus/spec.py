"""Runtime specification for ReFocus."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.refocus.model import Model

"""Validated parameters for ReFocus."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    # FLinear's irfft needs at least two output samples.
    d_model: int = Field(default=128, ge=2)
    d_pick: int = Field(default=32, ge=2)
    layers: int = Field(default=2, ge=1)
    dropout: float = 0.1
    beta: float = 0.5
    kernel_size: int = Field(default=25, ge=1)
    initial: bool = True

    @field_validator("kernel_size")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return value


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        d_pick=params["d_pick"],
        layers=params["layers"],
        dropout=params["dropout"],
        beta=params["beta"],
        kernel_size=params["kernel_size"],
        initial=params["initial"],
    )


SPEC = ModelSpec(
    name="ReFocus",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_refocus.toml",
    capabilities=frozenset({"time-series"}),
    components=("energy_frequency_pooling", "revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
