"""Model specification for MTSMixer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mtsmixer.model import Model

from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    d_ff: int = Field(default=4, gt=0)
    e_layers: int = Field(default=2, gt=0)
    fac_T: bool = True
    fac_C: bool = True
    sampling: int = Field(default=2, gt=0)
    norm: bool = True
    individual: bool = False
    rev: bool = True

    @model_validator(mode="after")
    def _channel_rank(self) -> "ModelParameterConfig":
        if self.fac_C and self.d_ff >= self.enc_in:
            raise ValueError("factorized channel rank must be smaller than enc_in")
        return self


def build_model(cfg, params):
    """Construct MTSMixer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        fac_T=bool(params["fac_T"]),
        fac_C=bool(params["fac_C"]),
        sampling=params["sampling"],
        norm=bool(params["norm"]),
        individual=bool(params["individual"]),
        rev=bool(params["rev"]),
    )


SPEC = ModelSpec(
    name="MTSMixer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("channel_wise_linear", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
