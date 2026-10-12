"""Model specification for STGCN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stgcn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGCN parameters supplied via ``model.params``.

    ``enc_in`` (the number of spatial nodes ``N``) is required. ``num_nodes``
    and ``adj_mx`` are injected by the runner from the dataset and are NOT
    declared in the TOML.
    """

    enc_in: int
    input_dim: int = 3
    Kt: int = 3
    Ks: int = 3
    hidden_dim: int = 64
    bottleneck_dim: int = 16
    out_hidden_dim: int = 128
    act_func: str = "glu"
    graph_conv_type: str = "cheb_graph_conv"
    bias: bool = True
    droprate: float = 0.5


def build_model(cfg, params):
    """Construct STGCN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        Kt=params["Kt"],
        Ks=params["Ks"],
        hidden_dim=params["hidden_dim"],
        bottleneck_dim=params["bottleneck_dim"],
        out_hidden_dim=params["out_hidden_dim"],
        act_func=params["act_func"],
        graph_conv_type=params["graph_conv_type"],
        bias=params["bias"],
        droprate=params["droprate"],
    )


SPEC = ModelSpec(
    name="STGCN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_spectral", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
