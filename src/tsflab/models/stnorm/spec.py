"""Model specification for STNorm."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stnorm.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STNorm parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    channels: int = 16
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    tnorm_bool: bool = True
    snorm_bool: bool = True


def build_model(cfg, params):
    """Construct STNorm from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        channels=params["channels"],
        kernel_size=params["kernel_size"],
        blocks=params["blocks"],
        layers=params["layers"],
        tnorm_bool=params["tnorm_bool"],
        snorm_bool=params["snorm_bool"],
    )


SPEC = ModelSpec(
    name="STNorm",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
