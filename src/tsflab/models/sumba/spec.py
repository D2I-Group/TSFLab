"""Runtime specification for Sumba."""

from typing import Annotated

from pydantic import BaseModel, Field
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sumba.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=32, gt=0)
    basis_count: int = Field(default=4, gt=0)
    basis_rank: int = Field(default=8, gt=0)
    temporal_kernels: list[Annotated[int, Field(ge=1)]] = Field(default=[2, 3, 5], min_length=1)
    depth: int = Field(default=2, gt=0)
    diffusion_steps: int = Field(default=2, gt=0)
    mix: float = Field(default=0.1, ge=0, le=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        d_model=params["d_model"],
        basis_count=params["basis_count"],
        basis_rank=params["basis_rank"],
        temporal_kernels=tuple(params["temporal_kernels"]),
        depth=params["depth"],
        diffusion_steps=params["diffusion_steps"],
        mix=params["mix"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="Sumba",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("instance_norm",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
