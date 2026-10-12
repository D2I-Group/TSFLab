"""Runtime specification for TimeGS."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timegs.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    periods: list[int] = Field(default_factory=lambda: [24, 24, 24], min_length=1)
    fold_width: int = Field(default=24, gt=0)
    image_height: int = Field(default=24, gt=0)
    image_width: int = Field(default=24, gt=0)
    hidden_dim: int = Field(default=16, gt=1)
    conv_dim: int = Field(default=8, gt=0)
    components: int = Field(default=2, gt=0)
    ngf: int = Field(default=4, gt=0)
    n_downsampling: int = Field(default=3, gt=0)
    n_blocks: int = Field(default=1, ge=0)
    kernel_size: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    rows: int = Field(default=7, gt=0)
    draft_len: int = Field(default=11, gt=0)
    ratio: int = Field(default=1, gt=0)
    extend_len: int = Field(default=0, ge=0)
    cholesky1: list[float] = Field(default_factory=lambda: [0.4, 0.8, 1.2], min_length=1)
    cholesky2: list[float] = Field(default_factory=lambda: [0.0, -0.2, 0.2], min_length=1)
    cholesky3: list[float] = Field(default_factory=lambda: [0.4, 0.8, 1.2], min_length=1)
    coefficient: list[float] = Field(default_factory=lambda: [0.5, 1.0], min_length=1)
    temperature: float = Field(default=0.1, gt=0.0)
    weight_norm: Literal["official", "paper"] = "official"
    mse_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _contract(self) -> ModelParameterConfig:
        step = 2**self.n_downsampling
        if self.image_height % step or self.image_width % step:
            raise ValueError(
                "image_height and image_width must be divisible by 2 ** n_downsampling"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep the image size")
        if any(p < 0 for p in self.periods):
            raise ValueError("periods must be non-negative (0 means the horizon span)")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_objective(model, batch, criterion):
    """Appendix B, Eq. (18): ``mse_weight * MSE + (1 - mse_weight) * MAE`` on the
    aligned forecast (replaces the configured criterion, as the official
    ``MSEMAE`` loss does)."""
    del criterion
    forecast = batch.forecast(model)
    return forecast, model.hybrid_loss(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="TimeGS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_timegs.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
