"""Runtime specification for EvoMSN."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.evomsn.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    top_k: int = Field(default=4, gt=0)
    periods: list[int] | None = None
    stat_hidden: int = Field(default=512, gt=0)
    stat_pretrain_epochs: int = Field(default=5, ge=0)
    stat_lr: float = Field(default=1e-4, gt=0)
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if self.periods is not None and (len(self.periods) != self.top_k or min(self.periods) < 1):
            raise ValueError("periods must list top_k positive integers")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="EvoMSN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_evomsn.toml",
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
