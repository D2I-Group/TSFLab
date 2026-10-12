"""Runtime specification for WDformer."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wdformer.model import Model

"""Validated parameters for WDformer."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = 256
    dropout: float = 0.1
    wave_size: int = Field(default=3, gt=0, le=12)

    @model_validator(mode="after")
    def _check_heads(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        wave_size=params["wave_size"],
    )


SPEC = ModelSpec(
    name="WDformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_wdformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("differential_attention", "instance_norm", "wavelet"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
