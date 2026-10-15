"""Model specification for DCRNN."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dcrnn.model import Model


class ModelParameterConfig(BaseModel):
    """Validated DCRNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int = Field(ge=1)
    input_dim: int = Field(default=3, ge=1)
    rnn_units: int = Field(default=16, ge=1)
    num_rnn_layers: int = Field(default=1, ge=1)
    max_diffusion_step: int = Field(default=2, ge=0)


def build_model(cfg, params):
    """Construct DCRNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        rnn_units=params["rnn_units"],
        num_rnn_layers=params["num_rnn_layers"],
        max_diffusion_step=params["max_diffusion_step"],
    )


SPEC = ModelSpec(
    name="DCRNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("channel_alignment", "graph_conv_gru", "graph_utils", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
