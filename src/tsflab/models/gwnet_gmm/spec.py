"""Runtime specification for GWNetGMM."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gwnet_gmm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` is injected by the runner from the dataset; the predefined
    ``adj_mx`` is accepted from the runner but unused (adaptive adjacency only).
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    in_dim: int = Field(default=2, ge=1, le=3)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    hidden_channels: int = Field(default=32, gt=0)
    skip_channels: int = Field(default=256, gt=0)
    end_channels: int = Field(default=512, gt=0)
    kernel_size: int = Field(default=2, ge=2)
    blocks: int = Field(default=4, gt=0)
    layers: int = Field(default=2, gt=0)
    node_embedding_dim: int = Field(default=10, gt=0)
    anchors: list[float] = Field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0], min_length=1)
    spacing: float = Field(default=1.0, gt=0.0)

    @field_validator("anchors")
    @classmethod
    def _ascending(cls, anchors: list[float]) -> list[float]:
        if any(b <= a for a, b in zip(anchors, anchors[1:])):
            raise ValueError("anchors must be strictly ascending")
        return anchors


def build_model(cfg, params):
    options = {key: value for key, value in params.items() if key not in ("enc_in", "num_nodes", "adj_mx")}
    options["anchors"] = tuple(options["anchors"])
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        **options,
    )


def training_objective(model, batch, criterion):
    """Train on the GMM negative log-likelihood (Eq. 5); validation uses the configured loss."""
    del criterion
    return model.training_objective(batch.x, batch.x_mark, batch.target)


SPEC = ModelSpec(
    name="GWNetGMM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gwnet_gmm.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("adaptive_node_embedding_adjacency", "diffusion_conv", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
