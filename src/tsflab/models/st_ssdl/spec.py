"""Model specification for ST-SSDL."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.st_ssdl.model import Model


class ModelParameterConfig(BaseModel):
    """Validated ST-SSDL parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset,
    so they are not declared here. ``enc_in`` (= number of nodes ``N``) is
    required; the remaining defaults are kept modest for a fast spatiotemporal
    smoke run.
    """

    enc_in: int
    input_dim: int = Field(default=3, gt=0)
    rnn_units: int = Field(default=16, gt=0)
    rnn_layers: int = Field(default=1, gt=0)
    cheb_k: int = Field(default=2, gt=0)
    prototype_num: int = Field(default=8, gt=0)
    prototype_dim: int = Field(default=8, gt=0)
    output_dim: int = Field(default=1, ge=1, le=1)  # TSFLab ST-SSDL exposes one value per node


def build_model(cfg, params):
    """Construct ST-SSDL from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        rnn_units=params["rnn_units"],
        rnn_layers=params["rnn_layers"],
        cheb_k=params["cheb_k"],
        prototype_num=params["prototype_num"],
        prototype_dim=params["prototype_dim"],
        output_dim=params["output_dim"],
    )


SPEC = ModelSpec(
    name="ST-SSDL",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("deviation_memory", "graph_conv_gru", "graph_utils", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
