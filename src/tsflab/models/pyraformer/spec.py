"""Model specification for Pyraformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.pyraformer.model import Model

from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    window_size: tuple[int, ...] = (4, 4)
    inner_size: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not self.window_size or any(factor < 2 for factor in self.window_size):
            raise ValueError("window_size must contain branching factors >= 2")
        if self.inner_size % 2 == 0:
            raise ValueError("inner_size must be odd")
        return self


def build_model(cfg, params):
    """Construct Pyraformer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        window_size=params["window_size"],
        inner_size=params["inner_size"],
    )


SPEC = ModelSpec(
    name="Pyraformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
