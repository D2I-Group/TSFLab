"""Model specification for DFDGCN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dfdgcn.model import Model

from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    """Validated DFDGCN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here; ``adj_mx`` is required.
    """

    enc_in: int
    dropout: float = 0.3
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    a: float = 1.0
    fft_emb: int = 10
    identity_emb: int = 10
    hidden_emb: int = 30
    subgraph: int = Field(default=20, ge=1)
    steps_per_day: int = Field(default=288, ge=1)

    @model_validator(mode="after")
    def _equal_widths(self):
        if self.residual_channels != self.dilation_channels:
            raise ValueError("local DFDGCN requires equal residual and dilation widths")
        return self


def build_model(cfg, params):
    """Construct DFDGCN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        dropout=params["dropout"],
        residual_channels=params["residual_channels"],
        dilation_channels=params["dilation_channels"],
        skip_channels=params["skip_channels"],
        end_channels=params["end_channels"],
        kernel_size=params["kernel_size"],
        blocks=params["blocks"],
        layers=params["layers"],
        a=params["a"],
        fft_emb=params["fft_emb"],
        identity_emb=params["identity_emb"],
        hidden_emb=params["hidden_emb"],
        subgraph=params["subgraph"],
        steps_per_day=params["steps_per_day"],
    )


SPEC = ModelSpec(
    name="DFDGCN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("adaptive_node_embedding_adjacency", "gated_dilated_conv", "graph_utils", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
)
