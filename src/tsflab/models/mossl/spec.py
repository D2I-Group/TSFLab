"""Runtime specification for MoSSL."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mossl.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``enc_in = num_modalities * num_nodes`` with modality-major channels
    (channel ``m * L + l``). The runner may inject ``num_nodes`` and ``adj_mx``;
    MoSSL uses no predefined graph and ignores them.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_modalities: int = Field(default=1, gt=0)
    hidden_channels: int = Field(default=48, ge=2)
    num_components: int = Field(default=4, gt=0)
    layers: int = Field(default=4, gt=0)
    kernel_size: int = Field(default=2, ge=2)
    mask_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    self_supervised: bool = True

    @model_validator(mode="after")
    def _modalities(self) -> "ModelParameterConfig":
        if self.enc_in % self.num_modalities:
            raise ValueError(
                f"enc_in={self.enc_in} must equal num_modalities * num_nodes (num_modalities={self.num_modalities})"
            )
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(
        **{key: value for key, value in params.items() if key not in ("num_nodes", "adj_mx")}
    ).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="MoSSL",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mossl.toml",
    capabilities=frozenset({"spatiotemporal"}),
    contract_task={"seq_len": 16, "pred_len": 3, "label_len": 0},
)
