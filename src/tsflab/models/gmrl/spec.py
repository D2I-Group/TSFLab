"""Runtime specification for GMRL."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gmrl.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``enc_in = num_sources * num_locations`` with source-major channels
    (channel ``s * L + l``). The runner may inject ``num_nodes`` and ``adj_mx``;
    GMRL uses no graph and ignores them.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_sources: int = Field(default=1, gt=0)
    hidden_channels: int = Field(default=24, gt=0)
    num_components: int = Field(default=17, gt=0)
    layers: int = Field(default=4, gt=0)
    kernel_size: int = Field(default=2, ge=2)
    memory_size: int = Field(default=8, gt=0)
    use_memory: bool = True
    cluster_weight: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _sources_divide_channels(self) -> "ModelParameterConfig":
        if self.enc_in % self.num_sources:
            raise ValueError(
                f"enc_in={self.enc_in} must equal num_sources * num_locations (num_sources={self.num_sources})"
            )
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(
        **{key: value for key, value in params.items() if key not in ("num_nodes", "adj_mx")}
    ).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="GMRL",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gmrl.toml",
    capabilities=frozenset({"spatiotemporal"}),
    contract_task={"seq_len": 16, "pred_len": 3, "label_len": 0},
)
