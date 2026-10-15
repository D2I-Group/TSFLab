"""Runtime specification for TimeCAP."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timecap.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=32, gt=0)
    patch_len: int = Field(default=16, gt=0)
    group_size: int = Field(default=4, gt=0)
    group_stride: int = Field(default=2, gt=0)
    num_heads: int = Field(default=4, gt=0)
    dropout: float = 0.1
    fusion_alpha: float = 0.1
    fusion_midpoint: float | None = None

    @model_validator(mode="after")
    def check_heads(self):
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        group_size=params["group_size"],
        group_stride=params["group_stride"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
        fusion_alpha=params["fusion_alpha"],
        fusion_midpoint=params.get("fusion_midpoint"),
    )


SPEC = ModelSpec(
    name="TimeCAP",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
