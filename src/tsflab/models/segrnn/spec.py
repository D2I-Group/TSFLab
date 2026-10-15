"""Model specification for SegRNN."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.segrnn.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    seg_len: int = Field(default=24, gt=0)

    @model_validator(mode="after")
    def check_even_width(self):
        if self.d_model % 2:
            raise ValueError("d_model must be even for relative/channel embeddings")
        return self


def build_model(cfg, params):
    """Construct SegRNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        seg_len=params["seg_len"],
    )


SPEC = ModelSpec(
    name="SegRNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("last_value_center",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
