"""Runtime specification for AMD."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.amd.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    n_block: int = Field(default=1, gt=0)
    alpha: float = 0.0
    mix_layer_num: int = Field(default=3, ge=0)
    mix_layer_scale: int = 2
    patch: int = Field(default=16, gt=0)
    dropout: float = 0.1
    num_experts: int = Field(default=8, gt=0)
    top_k: int = Field(default=2, ge=1)
    ff_dim: int = Field(default=2048, gt=0)
    moe_loss_coef: float = 1.0
    use_revin: bool = True
    batchnorm: bool = True

    @model_validator(mode="after")
    def _architecture(self) -> "ModelParameterConfig":
        if self.mix_layer_scale < 2 and self.mix_layer_num > 0:
            raise ValueError("mix_layer_num must be >= 0 and mix_layer_scale >= 2")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        n_block=params["n_block"],
        alpha=params["alpha"],
        mix_layer_num=params["mix_layer_num"],
        mix_layer_scale=params["mix_layer_scale"],
        patch=params["patch"],
        dropout=params["dropout"],
        num_experts=params["num_experts"],
        top_k=params["top_k"],
        ff_dim=params["ff_dim"],
        moe_loss_coef=params["moe_loss_coef"],
        use_revin=params["use_revin"],
        batchnorm=params["batchnorm"],
    )


SPEC = ModelSpec(
    name="AMD",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_amd.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
