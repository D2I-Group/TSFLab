"""Runtime specification for MultiSPANS."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.multispans.model import Model


def _check_kernels(conv_kernels) -> None:
    if any(k < 1 for k in conv_kernels):
        raise ValueError("conv_kernels must be positive")


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and the predefined ``adj_mx`` are injected by the runner from the
    dataset; without an adjacency the identity graph is used.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    mark_features: int = Field(default=0, ge=0)
    embed_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=3, gt=0)
    num_heads: int = Field(default=8, gt=0)
    conv_kernels: list[int] = Field(default_factory=lambda: [1, 2, 3, 6], min_length=1)
    conv_stride: int = Field(default=1, gt=0)
    gconv_hops: int = Field(default=3, ge=0)
    gconv_alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    att_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ffn_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    lape_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    spatial_masks: bool = True
    max_tree_height: int = Field(default=3, ge=1)
    position_routing: Literal["official", "matched"] = "official"

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        _check_kernels(self.conv_kernels)
        if (
            self.embed_dim % (len(self.conv_kernels) * (self.gconv_hops + 1))
            or self.embed_dim % self.num_heads
        ):
            raise ValueError(
                "embed_dim must be divisible by len(conv_kernels) * (gconv_hops + 1) and num_heads"
            )
        if self.spatial_masks and self.max_tree_height >= self.num_heads:
            raise ValueError(
                "num_heads must exceed the masked heads (max_tree_height - 1 levels + adjacency)"
            )
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(
        **{key: value for key, value in params.items() if key not in ("num_nodes", "adj_mx")}
    ).model_dump()
    enc_in = options.pop("enc_in")
    options["conv_kernels"] = tuple(options["conv_kernels"])
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", enc_in),
        adj_mx=params.get("adj_mx"),
        **options,
    )


SPEC = ModelSpec(
    name="MultiSPANS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_multispans.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("marks", "positional_encoding"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
