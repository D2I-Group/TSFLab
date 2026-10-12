"""Model specification for NBeats."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.nbeats.model import Model

from pydantic import BaseModel, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int
    stack_types: list[str] = ["trend", "seasonality", "generic"]
    nb_blocks_per_stack: int = 3
    thetas_dim: list[int] = [4, 8, 8]
    hidden_layer_units: int = 256
    share_weights_in_stack: bool = False
    nb_harmonics: int | None = None

    @model_validator(mode="after")
    def validate_stacks(self):
        if len(self.stack_types) != len(self.thetas_dim):
            raise ValueError("stack_types and thetas_dim must have equal length")
        return self


def build_model(cfg, params):
    """Construct NBeats from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        stack_types=tuple(params["stack_types"]),
        nb_blocks_per_stack=params["nb_blocks_per_stack"],
        thetas_dim=tuple(params["thetas_dim"]),
        hidden_layer_units=params["hidden_layer_units"],
        share_weights_in_stack=bool(params["share_weights_in_stack"]),
        nb_harmonics=params["nb_harmonics"],
    )


SPEC = ModelSpec(
    name="NBeats",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
