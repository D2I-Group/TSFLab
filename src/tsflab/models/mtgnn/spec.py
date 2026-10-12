"""Model specification for MTGNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mtgnn.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated MTGNN parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset and need not be
    declared in the TOML.
    """

    enc_in: int = Field(gt=0)
    input_dim: int = Field(default=3, gt=0)
    gcn_depth: int = Field(default=2, ge=0)
    subgraph_size: int = 20
    node_dim: int = 40
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    layers: int = Field(default=3, gt=0)
    dropout: float = 0.3
    propalpha: float = 0.05
    tanhalpha: float = 3.0
    dilation_exponential: int = 1
    build_adj: bool = True


def build_model(cfg, params):
    """Construct MTGNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        gcn_depth=params["gcn_depth"],
        subgraph_size=params["subgraph_size"],
        node_dim=params["node_dim"],
        conv_channels=params["conv_channels"],
        residual_channels=params["residual_channels"],
        skip_channels=params["skip_channels"],
        end_channels=params["end_channels"],
        layers=params["layers"],
        dropout=params["dropout"],
        propalpha=params["propalpha"],
        tanhalpha=params["tanhalpha"],
        dilation_exponential=params["dilation_exponential"],
        build_adj=params["build_adj"],
    )


SPEC = ModelSpec(
    name="MTGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("gated_dilated_conv", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
