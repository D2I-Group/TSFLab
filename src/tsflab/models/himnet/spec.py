"""Model specification for HimNet."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.himnet.model import Model


class ModelParameterConfig(BaseModel):
    """Validated HimNet parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in TOML. ``enc_in`` (= number of nodes ``N``) is the
    required channel count; the remaining fields carry modest defaults for a
    fast smoke run.
    """

    enc_in: int
    input_dim: int = 3
    output_dim: Literal[1] = 1  # TSFLab HimNet exposes one value per node
    hidden_dim: int = 32
    num_layers: int = 1
    cheb_k: int = 2
    node_embedding_dim: int = 8
    st_embedding_dim: int = 8
    tod_embedding_dim: int = 8
    dow_embedding_dim: int = 8
    steps_per_day: int = 288
    use_teacher_forcing: bool = True


def build_model(cfg, params):
    """Construct HimNet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        output_dim=params["output_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        cheb_k=params["cheb_k"],
        node_embedding_dim=params["node_embedding_dim"],
        st_embedding_dim=params["st_embedding_dim"],
        tod_embedding_dim=params["tod_embedding_dim"],
        dow_embedding_dim=params["dow_embedding_dim"],
        steps_per_day=params["steps_per_day"],
        use_teacher_forcing=params["use_teacher_forcing"],
    )


SPEC = ModelSpec(
    name="HimNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_conv_gru", "marks", "node_adaptive_graph_conv"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
