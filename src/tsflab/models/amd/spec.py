"""Runtime specification for AMD."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.amd.model import Model

"""Validated parameters for AMD."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    n_block: int = 1
    alpha: float = 0.0
    mix_layer_num: int = 3
    mix_layer_scale: int = 2
    patch: int = 16
    dropout: float = 0.1
    num_experts: int = 8
    top_k: int = 2
    ff_dim: int = 2048
    moe_loss_coef: float = 1.0
    use_revin: bool = True
    batchnorm: bool = True


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
