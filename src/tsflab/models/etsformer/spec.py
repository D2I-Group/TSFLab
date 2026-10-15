"""Model specification for ETSformer."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.etsformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = 2
    d_ff: int = 256
    top_k: int = 3
    dropout: float = 0.1
    activation: str = "sigmoid"

    @model_validator(mode="after")
    def _heads_divide_width(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    """Construct ETSformer from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_ff=params["d_ff"],
        top_k=params["top_k"],
        dropout=params["dropout"],
        activation=params["activation"],
    )


SPEC = ModelSpec(
    name="ETSformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
