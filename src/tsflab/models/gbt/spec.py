"""Runtime specification for GBT."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gbt.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    ar_len: int | None = Field(default=None, gt=0)
    fd_model: int = Field(default=32, gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    ar_blocks: int = Field(default=3, gt=0)
    pyramid: int = Field(default=3, gt=0)
    d_layers: int = Field(default=2, gt=0)
    kernel: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    channel_independent: bool = True
    fused_conv: bool = False
    mix: bool = True
    ar_epochs: int = Field(default=1, ge=0)
    ar_lr: float = Field(default=1e-4, gt=0.0)

    @model_validator(mode="after")
    def validate_shapes(self):
        if self.kernel % 2 == 0:
            raise ValueError("GBT kernel must be odd")
        if self.pyramid > self.ar_blocks:
            raise ValueError("GBT needs 1 <= pyramid <= ar_blocks")
        if self.fd_model % self.n_heads:
            raise ValueError("Auto-Regression width must be divisible by n_heads")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="GBT",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gbt.toml",
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    components=("embed", "masking"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
