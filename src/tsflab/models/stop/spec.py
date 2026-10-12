"""Model specification for STOP."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stop.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STOP parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 16
    prompt_dim: int = 16
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    core: int = 4
    head: int = 4


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
