"""Runtime specification for FiLM."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.film.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    multiscale: list[int] = Field(default_factory=lambda: [1, 2, 4], min_length=1)
    order: int = Field(default=64, gt=0)
    rank: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _low_rank_contract(self) -> "ModelParameterConfig":
        if self.rank > self.order:
            raise ValueError("rank cannot exceed order")
        if any(scale < 1 for scale in self.multiscale):
            raise ValueError("multiscale entries must be positive")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="FiLM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
