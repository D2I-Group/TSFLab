"""Runtime specification for HMformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.hmformer.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    dropout: float = 0.1
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    num_scales: int = Field(default=3, gt=0)
    depth: int = Field(default=1, gt=0)
    num_heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _even_head_dimension(self) -> "ModelParameterConfig":
        # Every scale doubles d_model, so the first scale's heads decide the rule.
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("each attention head must have a positive even dimension")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        dropout=params["dropout"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        num_scales=params["num_scales"],
        depth=params["depth"],
        num_heads=params["num_heads"],
    )


SPEC = ModelSpec(
    name="HMformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
