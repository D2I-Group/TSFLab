"""Model specification for AGCRN."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.agcrn.model import Model


class ModelParameterConfig(BaseModel):
    """Validated AGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    are therefore not declared here. ``enc_in`` (= number of nodes ``N``) is the
    only required field; the remaining defaults are kept modest for a fast
    spatiotemporal smoke.
    """

    enc_in: int
    input_dim: int = Field(default=3, gt=0)
    rnn_units: int = Field(default=32, gt=0)
    embed_dim: int = Field(default=8, gt=0)
    num_layers: int = Field(default=1, gt=0)
    cheb_k: int = Field(default=2, gt=0)
    output_dim: Literal[1] = 1  # TSFLab AGCRN exposes one value per node


def build_model(cfg, params):
    """Construct AGCRN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        rnn_units=params["rnn_units"],
        embed_dim=params["embed_dim"],
        num_layers=params["num_layers"],
        cheb_k=params["cheb_k"],
        output_dim=params["output_dim"],
    )


SPEC = ModelSpec(
    name="AGCRN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_conv_gru", "marks", "node_adaptive_graph_conv"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
