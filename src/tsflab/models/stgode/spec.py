"""Model specification for STGODE."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stgode.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGODE parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    hidden_dim: int = 32
    ode_steps: int = 2


def build_model(cfg, params):
    """Construct STGODE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        ode_steps=params["ode_steps"],
    )


SPEC = ModelSpec(
    name="STGODE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
