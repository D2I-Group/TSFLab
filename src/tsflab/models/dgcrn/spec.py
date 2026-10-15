"""Model specification for DGCRN."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dgcrn.model import Model


class ModelParameterConfig(BaseModel):
    """Validated DGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int = Field(ge=1)
    gcn_depth: int = Field(default=1, ge=1)
    rnn_size: int = Field(default=16, ge=1)
    node_dim: int = Field(default=8, ge=1)
    hyper_gnn_dim: int = Field(default=8, ge=1)
    middle_dim: int = Field(default=2, ge=1)
    tanhalpha: float = Field(default=3.0, gt=0)
    dropout: float = Field(default=0.3, ge=0, lt=1)


def build_model(cfg, params):
    """Construct DGCRN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        gcn_depth=params["gcn_depth"],
        rnn_size=params["rnn_size"],
        node_dim=params["node_dim"],
        hyper_gnn_dim=params["hyper_gnn_dim"],
        middle_dim=params["middle_dim"],
        tanhalpha=params["tanhalpha"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="DGCRN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_conv_gru", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
