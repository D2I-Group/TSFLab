"""Model specification for GWNet."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gwnet.model import Model

from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    """Validated GWNet parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = Field(default=3, gt=0)
    dropout: float = 0.3
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = Field(default=2, gt=0)
    layers: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def _residual_matches_dilation(self) -> "ModelParameterConfig":
        if self.residual_channels != self.dilation_channels:
            raise ValueError("local GWNet requires residual_channels == dilation_channels")
        return self


def build_model(cfg, params):
    """Construct GWNet from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        dropout=params["dropout"],
        residual_channels=params["residual_channels"],
        dilation_channels=params["dilation_channels"],
        skip_channels=params["skip_channels"],
        end_channels=params["end_channels"],
        kernel_size=params["kernel_size"],
        blocks=params["blocks"],
        layers=params["layers"],
    )


SPEC = ModelSpec(
    name="GWNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=(
        "adaptive_node_embedding_adjacency",
        "diffusion_conv",
        "gated_dilated_conv",
        "graph_utils",
        "marks",
    ),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
