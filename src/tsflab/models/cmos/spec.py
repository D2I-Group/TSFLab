"""Runtime specification for CMoS."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cmos.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    seg_size: int = Field(default=4, gt=0)
    num_map: int = Field(default=3, gt=0)
    kernel_size: int = Field(default=4, gt=0)
    period: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _period_matches_segments(self):
        if self.period is not None and (self.period < self.seg_size or self.period % self.seg_size):
            raise ValueError("period must be a multiple of seg_size")
        return self


def build_model(cfg, params):
    return Model(
        c_in=params["enc_in"],
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        seg_size=params["seg_size"],
        num_map=params["num_map"],
        kernel_size=params["kernel_size"],
        period=params.get("period"),
    )


SPEC = ModelSpec(
    name="CMoS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
