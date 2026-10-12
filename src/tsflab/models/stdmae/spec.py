"""Runtime specification for STDMAE."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stdmae.model import Model, receptive_field


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    history_len: int = Field(default=12, gt=0)
    patch_size: int = Field(default=12, gt=0)
    embed_dim: int = Field(default=96, gt=0)
    num_heads: int = Field(default=4, gt=0)
    mlp_ratio: int = Field(default=4, gt=0)
    encoder_depth: int = Field(default=4, gt=0)
    encoder_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    in_dim: int = Field(default=2, ge=1, le=3)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    residual_channels: int = Field(default=32, gt=0)
    dilation_channels: int = Field(default=32, gt=0)
    skip_channels: int = Field(default=256, gt=0)
    end_channels: int = Field(default=512, gt=0)
    kernel_size: int = Field(default=2, gt=0)
    blocks: int = Field(default=4, gt=0)
    layers: int = Field(default=2, gt=0)
    freeze_encoders: bool = False

    @model_validator(mode="after")
    def _architecture_contract(self):
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        receptive = receptive_field(self.blocks, self.layers, self.kernel_size)
        if self.history_len + 1 > receptive:
            raise ValueError(f"history_len + 1 must not exceed the receptive field {receptive}")
        return self


def build_model(cfg, params):
    options = {
        key: value for key, value in params.items() if key not in ("enc_in", "num_nodes", "adj_mx")
    }
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        **options,
    )


SPEC = ModelSpec(
    name="STDMAE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_stdmae.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=(
        "adaptive_node_embedding_adjacency",
        "diffusion_conv",
        "graph_utils",
        "marks",
        "tst_transformer",
    ),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
