"""Model specification for STOP."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stop.model import Model


class ModelParameterConfig(BaseModel):
    """Validated STOP parameters supplied via ``model.params``."""

    enc_in: int = Field(gt=0)
    model_dim: int = Field(default=16, gt=0)
    prompt_dim: int = Field(default=16, gt=0)
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    core: int = Field(default=4, gt=0)
    head: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _heads(self) -> ModelParameterConfig:
        if (self.model_dim + 2 * self.prompt_dim) % self.head:
            raise ValueError("STOP model_dim + 2*prompt_dim must be divisible by head")
        return self


def build_model(cfg, params):
    """Construct STOP from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        model_dim=params["model_dim"],
        prompt_dim=params["prompt_dim"],
        num_layer=params["num_layer"],
        hid_dim=params["hid_dim"],
        tod_size=params["tod_size"],
        kernel_size=params["kernel_size"],
        core=params["core"],
        head=params["head"],
    )


SPEC = ModelSpec(
    name="STOP",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks", "series_decomposition"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
