"""Model specification for NHiTS."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.nhits.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int
    # Local N-HiTS supports the paper identity interpolation basis only.
    stack_types: list[Literal["identity"]] = ["identity", "identity", "identity"]
    n_blocks: list[int] = [1, 1, 1]
    mlp_units: list = [[256, 256]]
    n_pool_kernel_size: list[int] = [2, 2, 1]
    n_freq_downsample: list[int] = [4, 2, 1]
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    dropout: float = 0.0
    activation: str = "ReLU"
    use_norm: bool = True

    @model_validator(mode="after")
    def _stack_lengths(self):
        count = len(self.stack_types)
        if not all(len(items) == count for items in (self.n_blocks, self.n_pool_kernel_size, self.n_freq_downsample)):
            raise ValueError("N-HiTS stack configuration lengths must match")
        return self


def build_model(cfg, params):
    """Construct NHiTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        enc_in=params["enc_in"],
        stack_types=params["stack_types"],
        n_blocks=params["n_blocks"],
        mlp_units=params["mlp_units"],
        n_pool_kernel_size=params["n_pool_kernel_size"],
        n_freq_downsample=params["n_freq_downsample"],
        pooling_mode=params["pooling_mode"],
        interpolation_mode=params["interpolation_mode"],
        dropout=params["dropout"],
        activation=params["activation"],
        use_norm=bool(params["use_norm"]),
    )


SPEC = ModelSpec(
    name="NHiTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
